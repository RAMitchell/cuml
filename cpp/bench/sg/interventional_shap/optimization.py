# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated research backends. No changes to public TreeExplainer dispatch."""

import math
from pathlib import Path

import cupy as cp
import numpy as np

NODE = np.dtype(
    dict(
        names=[
            "left",
            "right",
            "feature",
            "value",
            "begin",
            "end",
            "missing",
            "le",
            "categorical",
            "right_categories",
        ],
        formats=["i4", "i4", "i4", "f8", "u8", "u8", "?", "?", "?", "?"],
        offsets=[0, 4, 8, 16, 24, 32, 40, 41, 42, 43],
        itemsize=48,
    )
)
VARIANTS = {
    "control": {},
    "lookup_global": {"WEIGHTS": 1},
    "lookup_shared": {"WEIGHTS": 2},
    "lookup_constant": {"WEIGHTS": 3},
    "incremental": {"WEIGHTS": 4},
    "cache_background": {"CACHE_R": 1},
    "cache_foreground": {"CACHE_X": 1},
    "cache_both": {"CACHE_R": 1, "CACHE_X": 1},
    "threshold_bins": {"BINS": 1},
    "quadrature_leaf": {"WEIGHTS": 5},
    "quadrature_subtree": {},
    "local_reduce": {"REDUCE": 1},
    "lookup_reduce": {"WEIGHTS": 1, "REDUCE": 1},
    "lookup_cache_reduce": {
        "WEIGHTS": 1,
        "REDUCE": 1,
        "CACHE_X": 1,
        "CACHE_R": 1,
    },
}


def launch(kernel, n, args, threads=128):
    if n:
        kernel(
            (min((int(n) + threads - 1) // threads, 65535),), (threads,), args
        )


class Model:
    def __init__(self, model):
        h = model.get_header_accessor()
        assert int(h.get_field("num_target")[0]) == 1
        self.groups = int(h.get_field("num_class")[0])
        self.bias = cp.asarray(h.get_field("base_scores"), dtype=cp.float64)
        self.divisor = (
            model.num_tree if h.get_field("average_tree_output")[0] else 1
        )
        vector = int(h.get_field("leaf_vector_shape")[1]) > 1
        class_ids = h.get_field("class_id")
        all_nodes = []
        all_cats = []
        roots = []
        groups = []
        offset = 0
        coffset = 0
        self.depth = 0
        for ti in range(model.num_tree):
            a = model.get_tree_accessor(ti)
            left = a.get_field("cleft")
            right = a.get_field("cright")
            copies = self.groups if vector else 1
            if offset + len(left) * copies > np.iinfo(np.int32).max:
                raise ValueError(
                    "experimental node indices exceed int32 capacity"
                )
            n = np.zeros(len(left), dtype=NODE)
            n["left"] = np.where(left < 0, -1, left + offset)
            n["right"] = np.where(right < 0, -1, right + offset)
            n["feature"] = a.get_field("split_index")
            n["feature"][left < 0] = -1
            n["value"] = np.where(
                left < 0, a.get_field("leaf_value"), a.get_field("threshold")
            )
            n["missing"] = a.get_field("default_left")
            n["le"] = a.get_field("cmp") == 3
            numeric = (left >= 0) & (a.get_field("node_type") == 1)
            assert np.isin(a.get_field("cmp")[numeric], [2, 3]).all()
            n["categorical"] = a.get_field("node_type") == 2
            n["right_categories"] = a.get_field("category_list_right_child")
            cats = a.get_field("category_list").copy()
            begin = a.get_field("category_list_begin")
            end = a.get_field("category_list_end")
            for nid in np.flatnonzero(n["categorical"]):
                cats[begin[nid] : end[nid]].sort()
            n["begin"] = begin + coffset
            n["end"] = end + coffset
            all_cats.append(cats)
            coffset += len(cats)
            pending = [(0, 0)]
            while pending:
                nid, d = pending.pop()
                self.depth = max(self.depth, d)
                if left[nid] >= 0:
                    pending.extend(
                        [(int(left[nid]), d + 1), (int(right[nid]), d + 1)]
                    )
            if vector:
                values = a.get_field("leaf_vector")
                vb = a.get_field("leaf_vector_begin")
                for g in range(self.groups):
                    ng = n.copy()
                    valid = left >= 0
                    ng["left"][valid] += g * len(n)
                    ng["right"][valid] += g * len(n)
                    for nid in np.flatnonzero(left < 0):
                        ng["value"][nid] = values[vb[nid] + g]
                    roots.append(offset + g * len(n))
                    groups.append(g)
                    all_nodes.append(ng)
                offset += len(n) * self.groups
            else:
                roots.append(offset)
                groups.append(int(class_ids[ti]))
                all_nodes.append(n)
                offset += len(n)
        assert self.depth <= 128
        self.depth = max(1, self.depth)
        self.host = np.empty(sum(len(n) for n in all_nodes), dtype=NODE)
        pos = 0
        for n in all_nodes:
            self.host[pos : pos + len(n)] = n
            pos += len(n)
        self.nodes = cp.asarray(self.host.view(np.uint8))
        self.roots = cp.asarray(roots, dtype=cp.int64)
        self.host_roots = roots
        self.groups_array = cp.asarray(groups, dtype=cp.int32)
        self.cats = cp.asarray(np.concatenate(all_cats), dtype=cp.uint32)
        self.ranks = None
        self.thresholds = None
        self.nbytes = sum(
            a.nbytes
            for a in [
                self.nodes,
                self.roots,
                self.groups_array,
                self.cats,
                self.bias,
            ]
        )

    def prepare_bins(self, nf):
        if self.ranks is not None:
            return
        n = self.host
        ranks = np.zeros(len(n), dtype=np.uint32)
        self.thresholds = []
        for f in range(nf):
            indices = np.flatnonzero((n["feature"] == f) & ~n["categorical"])
            thresholds = np.unique(n["value"][indices])
            ranks[indices] = np.searchsorted(thresholds, n["value"][indices])
            self.thresholds.append(cp.asarray(thresholds))
        self.ranks = cp.asarray(ranks)


def coefficients(depth):
    values = []
    for n in range(depth + 1):
        values.extend(1 / math.comb(n, a) for a in range(n // 2 + 1))
    return np.array(values), np.array(
        [0] + [1 / n for n in range(1, depth + 1)]
    )


class Traversal:
    def __init__(
        self, model, variant, dtype=np.float32, profile=False, fixed_quad=None
    ):
        self.model = model
        self.variant = variant
        self.settings = VARIANTS[variant]
        depth = next(d for d in [8, 16, 32, 64, 128] if model.depth <= d)
        # Specialize stack size but choose quadrature order from actual depth.
        nq = fixed_quad or max(1, (model.depth + 1) // 2)
        qwidth = min(32, 1 << (nq - 1).bit_length())
        table, recip = coefficients(depth)
        self.table = cp.asarray(table)
        self.recip = cp.asarray(recip)
        p, w = np.polynomial.legendre.leggauss(nq)
        p = (p + 1) / 2
        w = w / 2

        def array(name, data):
            return (
                f"__device__ __constant__ double {name}[{len(data)}]={{"
                + ",".join(float(v).hex() for v in data)
                + "};\n"
            )

        source = (
            array("ctable", table)
            + array("crecip", recip)
            + array("qt", p)
            + array("qw", w)
        )
        defaults = dict(
            WEIGHTS=0,
            CACHE_X=0,
            CACHE_R=0,
            BINS=0,
            REDUCE=0,
            PROFILE=int(profile),
        )
        defaults.update(self.settings)
        opts = [
            "--std=c++17",
            f"-DDEPTH={depth}",
            f"-DTABLE_SIZE={len(table)}",
            f"-DNQUAD={nq}",
            f"-DQWIDTH={qwidth}",
            f"-DDATA={'float' if np.dtype(dtype) == np.float32 else 'double'}",
        ]
        opts.extend(f"-D{k}={v}" for k, v in defaults.items())
        source += (
            Path(__file__).with_name("optimization_kernels.cu").read_text()
        )
        self.module = cp.RawModule(code=source, options=tuple(opts))
        self.kernel = self.module.get_function(
            "subtree" if variant == "quadrature_subtree" else "pairs"
        )
        self.pack = self.module.get_function("pack_decisions")
        self.finish = self.module.get_function("finish")
        self.counters = cp.zeros(3, dtype=cp.uint64)
        self.qwidth = qwidth
        self.nq = nq
        self.attributes = self.kernel.attributes
        self.empty = cp.empty(1, dtype=cp.uint32)
        self.px = self.pr = self.bx = self.br = self.empty
        self.cache_bytes = 0

    def bits(self, data):
        m = self.model
        nr, nf = data.shape
        out = cp.empty((len(m.host), (nr + 31) // 32), dtype=cp.uint32)
        launch(
            self.pack,
            out.size * 32,
            (
                m.nodes,
                np.int64(len(m.host)),
                m.cats,
                data,
                np.int64(nr),
                np.int32(nf),
                out,
            ),
        )
        return out

    def bins(self, data):
        self.model.prepare_bins(data.shape[1])
        out = cp.empty((*data.shape, 2), dtype=cp.uint32)
        for f, thresholds in enumerate(self.model.thresholds):
            for j, side in enumerate(["left", "right"]):
                out[:, f, j] = cp.where(
                    cp.isnan(data[:, f]),
                    np.uint32(0xFFFFFFFF),
                    cp.searchsorted(thresholds, data[:, f], side=side),
                )
        return out

    def prepare(self, x, r):
        self.px = self.pr = self.bx = self.br = self.empty
        if self.settings.get("CACHE_X"):
            self.px = self.bits(x)
        if self.settings.get("CACHE_R"):
            self.pr = self.bits(r)
        if self.settings.get("BINS"):
            self.bx = self.bins(x)
            self.br = self.bins(r)
        self.cache_bytes = sum(
            a.nbytes
            for a in [self.px, self.pr, self.bx, self.br]
            if a is not self.empty
        )

    def compute(self, x, r, out=None):
        nx, nf = x.shape
        nr = len(r)
        m = self.model
        if not nr:
            raise ValueError("nonempty background required")
        if out is None:
            out = cp.empty((nx, m.groups, nf + 1), dtype=cp.float64)
        out.fill(0)
        args = (
            m.nodes,
            m.roots,
            m.groups_array,
            np.int32(len(m.roots)),
            m.cats,
            x,
            np.int64(nx),
            np.int32(nf),
            r,
            np.int64(nr),
            np.int32(m.groups),
            out,
        )
        tasks = len(m.roots) * nx * nr
        if self.variant == "quadrature_subtree":
            launch(
                self.kernel,
                tasks
                * self.qwidth
                * ((self.nq + self.qwidth - 1) // self.qwidth),
                args,
            )
        else:
            launch(
                self.kernel,
                tasks,
                args
                + (
                    self.table,
                    self.recip,
                    self.px,
                    self.pr,
                    self.bx,
                    self.br,
                    m.ranks if m.ranks is not None else self.empty,
                    self.counters,
                ),
            )
        launch(
            self.finish,
            out.size,
            (
                out,
                np.int64(out.size),
                np.int32(nf),
                m.bias,
                np.int32(m.groups),
                np.float64(m.divisor),
            ),
        )
        return out


PATTERN_VARIANTS = [
    "pattern_background",
    "pattern_foreground",
    "pattern_sparse",
    "pattern_dense",
    "pattern_adaptive",
]


class Paths:
    def __init__(self, model):
        paths = []
        slots = []
        features = []
        lengths = []
        widths = []
        values = []
        groups = []
        nodes = model.host
        for root, group in zip(
            model.host_roots, cp.asnumpy(model.groups_array)
        ):
            pending = [(root, [], [], [])]
            while pending:
                nid, path, slot, fs = pending.pop()
                n = nodes[nid]
                if n["feature"] < 0:
                    paths.append(path)
                    slots.append(slot)
                    features.append(fs)
                    lengths.append(len(path))
                    widths.append(len(fs))
                    values.append(n["value"])
                    groups.append(group)
                else:
                    f = int(n["feature"])
                    newfs = fs if f in fs else fs + [f]
                    index = newfs.index(f)
                    pending.append(
                        (
                            int(n["right"]),
                            path + [-nid - 1],
                            slot + [index],
                            newfs,
                        )
                    )
                    pending.append(
                        (int(n["left"]), path + [nid], slot + [index], newfs)
                    )
        self.depth = max(1, max(lengths))
        self.widths = np.array(widths, dtype=np.int32)
        if max(widths) > 64:
            raise ValueError(
                "pattern prototypes support at most 64 distinct path features"
            )

        def padded(rows):
            out = np.zeros((len(rows), self.depth), dtype=np.int32)
            for i, row in enumerate(rows):
                out[i, : len(row)] = row
            return out

        self.path = padded(paths)
        self.slot = padded(slots)
        self.features = padded(features)
        self.lengths = np.array(lengths, dtype=np.int32)
        self.values = np.array(values)
        self.groups = np.array(groups, dtype=np.int32)


class Patterns:
    def __init__(
        self,
        model,
        paths,
        variant,
        dtype=np.float32,
        tile=64,
        memory_limit=256 * 1024**2,
    ):
        self.model = model
        self.paths = paths
        self.variant = variant
        self.tile = tile
        self.memory_limit = memory_limit
        source = (
            Path(__file__)
            .with_name("optimization_kernels.cu")
            .read_text()
            .split('extern "C" __global__ void pack_decisions')[0]
        )
        source += Path(__file__).with_name("pattern_kernels.cu").read_text()
        self.module = cp.RawModule(
            code=source,
            options=(
                "--std=c++17",
                f"-DDATA={'float' if np.dtype(dtype) == np.float32 else 'double'}",
            ),
        )
        self.kernels = {
            name: self.module.get_function(name)
            for name in [
                "patterns",
                "compress",
                "sparse_values",
                "scatter_sparse",
                "histogram",
                "zeta",
                "diagonal",
                "scatter_dense",
            ]
        }
        self.finalizer = Traversal(model, "control", dtype)
        self.tiles = []
        self.cache_bytes = 0
        self.choices = {}
        self.max_tile_bytes = 0

    def mask(self, tile, data):
        m = self.model
        p = self.paths
        nl = len(tile["widths"])
        nx, nf = data.shape
        out = cp.empty((nl, nx), dtype=cp.uint64)
        launch(
            self.kernels["patterns"],
            out.size,
            (
                m.nodes,
                m.cats,
                tile["path"],
                tile["slot"],
                tile["lengths"],
                tile["widths"],
                np.int32(nl),
                np.int32(p.depth),
                data,
                np.int32(nx),
                np.int32(nf),
                out,
            ),
        )
        return out

    def unique(self, mask):
        sorted_mask = cp.sort(mask, axis=1)
        nl, nr = mask.shape
        keys = cp.zeros_like(mask)
        counts = cp.zeros(mask.shape, dtype=cp.int32)
        sizes = cp.empty(nl, dtype=cp.int32)
        launch(
            self.kernels["compress"],
            nl,
            (sorted_mask, np.int32(nl), np.int32(nr), keys, counts, sizes),
        )
        return keys, counts, sizes

    def build_tile(self, indices, x, r):
        p = self.paths
        nl = len(indices)
        nx, nf = x.shape
        nr = len(r)
        k = int(p.widths[indices[0]])
        t = {
            name: cp.asarray(getattr(p, name)[indices])
            for name in [
                "path",
                "slot",
                "lengths",
                "widths",
                "features",
                "groups",
                "values",
            ]
        }
        xm = self.mask(t, x)
        rm = self.mask(t, r)
        t["xm"] = xm
        mode = self.variant
        if mode == "pattern_adaptive":
            # Exact dispatch: compare conservative work estimates for this tile.
            # Dense requires k transforms for k features; sparse uses observed keys.
            ux, _, us = self.unique(xm)
            ur, rc, rs = self.unique(rm)
            sparse_work = float(
                cp.sum(us.astype(cp.float64) * rs).get()
            ) * max(1, k)
            dense_work = (
                nl * (2**k) * max(1, k) ** 2 if k <= 20 else float("inf")
            )
            mode = (
                "pattern_dense"
                if k > 0 and dense_work < sparse_work
                else "pattern_sparse"
            )
            t["decision"] = {
                "sparse_work": sparse_work,
                "dense_work": dense_work,
            }
        self.choices[mode] = self.choices.get(mode, 0) + nl
        t["mode"] = mode
        if mode == "pattern_dense" and k > 0:
            if k > 20:
                raise ValueError(
                    "dense prototype limited to 20 unique features"
                )
            size = 1 << k
            f = cp.zeros((nl, size), dtype=cp.float64)
            launch(
                self.kernels["histogram"],
                rm.size,
                (rm, np.int32(nr), np.int32(nl), np.int32(size), f),
            )
            for bit in range(k):
                launch(
                    self.kernels["zeta"],
                    f.size,
                    (
                        f,
                        np.int64(f.size),
                        np.int32(size),
                        np.int32(1 << bit),
                        np.int32(0),
                    ),
                )
            s = cp.empty((nl, k, size), dtype=cp.float64)
            launch(
                self.kernels["diagonal"],
                s.size,
                (f, t["values"], np.int32(nl), np.int32(k), np.int32(size), s),
            )
            for bit in range(k):
                launch(
                    self.kernels["zeta"],
                    s.size,
                    (
                        s,
                        np.int64(s.size),
                        np.int32(size),
                        np.int32(1 << bit),
                        np.int32(1),
                    ),
                )
            t.update(f=f, s=s, k=k, size=size)
        else:
            grouped = mode != "pattern_background"
            if grouped:
                ux, _, us = self.unique(xm)
            else:
                ux = xm
                us = cp.full(nl, nx, dtype=cp.int32)
            if mode == "pattern_foreground":
                ur = rm
                rc = cp.ones(rm.shape, dtype=cp.int32)
                rs = cp.full(nl, nr, dtype=cp.int32)
            else:
                ur, rc, rs = self.unique(rm)
            values = cp.zeros((nl, nx, p.depth + 1), dtype=cp.float64)
            launch(
                self.kernels["sparse_values"],
                nl * nx,
                (
                    ux,
                    np.int32(nx),
                    us,
                    ur,
                    np.int32(nr),
                    rc,
                    rs,
                    t["widths"],
                    t["values"],
                    np.int32(nl),
                    np.int32(p.depth),
                    values,
                    np.int32(nr),
                ),
            )
            t.update(keys=ux, sizes=us, contributions=values, grouped=grouped)
            t["mode"] = "pattern_sparse"
        # Track all retained arrays; temporary sorting buffers are captured by pool peak measurements.
        t["bytes"] = sum(
            a.nbytes for a in t.values() if isinstance(a, cp.ndarray)
        )
        self.max_tile_bytes = max(self.max_tile_bytes, t["bytes"])
        return t

    def iter_indices(self, nx):
        p = self.paths
        for k in np.unique(p.widths):
            ids = np.flatnonzero(p.widths == k)
            estimate = (
                8 * nx * (p.depth + 2) + 8 * (2 ** int(k)) * (int(k) + 1)
                if k <= 20
                else 8 * nx * (p.depth + 2)
            )
            count = max(
                1, min(self.tile, self.memory_limit // max(1, estimate))
            )
            for begin in range(0, len(ids), count):
                yield ids[begin : begin + count]

    def prepare(self, x, r):
        if not len(r):
            raise ValueError("nonempty background required")
        self.tiles = []
        self.cache_bytes = 0
        self.choices = {}
        self.max_tile_bytes = 0
        self.direct = self.variant == "pattern_adaptive" and (
            len(x) <= 8 or len(r) <= 4
        )
        if self.direct:
            self.choices = {"direct": 1}
            self.finalizer.prepare(x, r)
            return
        for indices in self.iter_indices(len(x)):
            t = self.build_tile(indices, x, r)
            self.cache_bytes += t["bytes"]
            if self.cache_bytes > self.memory_limit:
                self.tiles = []
                raise MemoryError(
                    "persistent pattern cache exceeds configured memory budget; use compute_streamed"
                )
            self.tiles.append(t)

    def scatter(self, t, x, out):
        p = self.paths
        m = self.model
        nx, nf = x.shape
        nl = len(t["widths"])
        if t["mode"] == "pattern_dense":
            launch(
                self.kernels["scatter_dense"],
                nl * nx,
                (
                    t["xm"],
                    np.int32(nx),
                    t["s"],
                    t["f"],
                    t["values"],
                    t["features"],
                    t["groups"],
                    np.int32(nl),
                    np.int32(t["k"]),
                    np.int32(p.depth),
                    np.int32(t["size"]),
                    np.int32(nf),
                    np.int32(m.groups),
                    out,
                ),
            )
        else:
            launch(
                self.kernels["scatter_sparse"],
                nl * nx,
                (
                    t["xm"],
                    np.int32(nx),
                    t["keys"],
                    t["sizes"],
                    t["contributions"],
                    t["features"],
                    t["widths"],
                    t["groups"],
                    np.int32(nl),
                    np.int32(p.depth),
                    np.int32(nf),
                    np.int32(m.groups),
                    out,
                    np.int32(t["grouped"]),
                ),
            )

    def compute(self, x, r, out=None, streamed=False):
        if not len(r):
            raise ValueError("nonempty background required")
        m = self.model
        nx, nf = x.shape
        if out is None:
            out = cp.empty((nx, m.groups, nf + 1), dtype=cp.float64)
        out.fill(0)
        if self.variant == "pattern_adaptive" and (len(x) <= 8 or len(r) <= 4):
            self.choices = {"direct": 1}
            return self.finalizer.compute(x, r, out)
        if streamed:
            self.choices = {}
            self.max_tile_bytes = 0
            for indices in self.iter_indices(nx):
                self.scatter(self.build_tile(indices, x, r), x, out)
        else:
            for t in self.tiles:
                self.scatter(t, x, out)
        launch(
            self.finalizer.finish,
            out.size,
            (
                out,
                np.int64(out.size),
                np.int32(nf),
                m.bias,
                np.int32(m.groups),
                np.float64(m.divisor),
            ),
        )
        return out

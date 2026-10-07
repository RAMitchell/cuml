# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise optimization variants on the original independent oracle fixtures."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from experiment import exhaustive, validate
from optimization import (
    Model,
    Paths,
    Patterns,
    Traversal,
    VARIANTS,
    PATTERN_VARIANTS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS) + PATTERN_VARIANTS
    )
    args = parser.parse_args()
    records = []

    def check(name, model, x, r, expected):
        reference = exhaustive(model, x, r) if expected is None else expected
        m = Model(model)
        dx = cp.asarray(x)
        dr = cp.asarray(r)
        cp.cuda.Stream.null.synchronize()
        p = None
        for variant in args.variants:
            if variant in PATTERN_VARIANTS:
                if p is None:
                    try:
                        p = Paths(m)
                    except ValueError:
                        records.append(
                            dict(
                                case=name,
                                variant=variant,
                                status="unsupported",
                                reason="more than 64 distinct features",
                            )
                        )
                        continue
                if variant == "pattern_dense" and p.widths.max() > 20:
                    records.append(
                        dict(
                            case=name,
                            variant=variant,
                            status="unsupported",
                            reason="dense exponential size above 20 distinct features",
                        )
                    )
                    continue
                b = Patterns(m, p, variant, dtype=x.dtype)
            else:
                b = Traversal(m, variant, dtype=x.dtype)
            stream = cp.cuda.Stream(non_blocking=True)
            # Module construction/uploads complete before using the independent stream.
            cp.cuda.Stream.null.synchronize()
            with stream:
                b.prepare(dx, dr)
                actual = b.compute(dx, dr)
            stream.synchronize()
            actual = cp.asnumpy(actual)
            np.testing.assert_allclose(
                actual,
                reference,
                atol=2e-6,
                rtol=2e-6,
                err_msg=f"{name}/{variant}",
            )
            if variant in PATTERN_VARIANTS:
                with stream:
                    again = b.compute(dx, dr, streamed=True)
                stream.synchronize()
                np.testing.assert_allclose(
                    cp.asnumpy(again),
                    reference,
                    atol=2e-6,
                    rtol=2e-6,
                    err_msg=f"{name}/{variant}/streamed",
                )
            with stream:
                b.prepare(dx[:0], dr)
                empty = b.compute(dx[:0], dr)
                assert empty.shape == (0, m.groups, x.shape[1] + 1)
                try:
                    b.compute(dx, dr[:0])
                except ValueError:
                    pass
                else:
                    raise AssertionError("empty background must be rejected")
            stream.synchronize()
            records.append(
                dict(
                    case=name,
                    variant=variant,
                    status="passed",
                    max_error=float(np.max(np.abs(actual - reference))),
                    depth=m.depth,
                )
            )
            print(name, variant, "PASS", flush=True)
            del b
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")

    validate(None, include_legacy=False, check_callback=check)
    print(
        f"Validated {sum(r['status'] == 'passed' for r in records)} optimization checks",
        flush=True,
    )


if __name__ == "__main__":
    main()

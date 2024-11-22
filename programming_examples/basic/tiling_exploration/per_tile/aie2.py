# tiling_exploration/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D


def generate_module(
    tensor_height, tensor_width, tile_height, tile_width, generate_access_map=False
):
    # define types
    dtype = np.int32
    tensor_size = tensor_height * tensor_width
    tile_size = tile_height * tile_width
    flattened_tensor = np.ndarray[(tensor_size,), np.dtype[dtype]]
    flattened_tile = np.ndarray[(tile_size,), np.dtype[dtype]]

    tiler = TensorTiler2D.simple_tiler(
        (tensor_height, tensor_width), (tile_height, tile_width)
    )
    if generate_access_map:
        tiler.visualize(file_path="per_tile.png")
        return

    @device(AIEDevice.npu1_1col)
    def device_body():
        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, flattened_tile)

        # Set up compute tiles

        # Compute tile 2
        @core(ComputeTile2)
        def core_body():
            # TODO: better way to get mutable constant than buffer??
            access_counter = buffer(
                ComputeTile2,
                np.ndarray[(1,), np.dtype[dtype]],
                "access_counter",
                initial_value=np.array([0], dtype=dtype),
            )
            for _ in range_(sys.maxsize):
                elemOut = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(tile_size):
                    elemOut[i] = access_counter[0]
                    access_counter[0] += 1
                of_out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(flattened_tensor)
        def sequence(access_count):
            for t in tiler:
                out_task = shim_dma_single_bd_task(
                    of_out, access_count, tap=t, issue_token=True
                )
                dma_start_task(out_task)
                dma_await_task(out_task)


def main(opts):
    with mlir_mod_ctx() as ctx:
        generate_module(
            opts.tensor_height,
            opts.tensor_width,
            opts.tile_height,
            opts.tile_width,
            opts.generate_access_map,
        )
        if not opts.generate_access_map:
            print(ctx.module)


def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--tensor-height", required=True, help="Tensor height", type=int)
    p.add_argument("--tensor-width", required=True, help="Tensor width", type=int)
    p.add_argument("--tile-height", required=True, help="Tile height", type=int)
    p.add_argument("--tile-width", required=True, help="Tile width", type=int)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="Produce a file showing data access order",
    )
    return p


if __name__ == "__main__":
    p = get_arg_parser()
    opts = p.parse_args()
    main(opts)

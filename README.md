# worker-comfyui-wan-refine

RunPod serverless worker for **video refinement**: takes a draft mp4 (typically 480×832 / 16fps from `worker-comfyui-wan`), runs face restore + 4× upscale + downscale to 1080p + RIFE 16→32fps + h264 CRF 18 encode, and uploads the finished mp4 to the network volume.

Forked from `dpoliyivets/worker-comfyui-wan` so the two workers can evolve independently — the base video gen image stays small and frozen on a known-good Wan2.2 stack while this image carries the additional refinement deps (face restore + RIFE + extra upscale model).

## What this worker does

- Receives a video file path (on the mounted network volume) + a ComfyUI workflow JSON.
- Loads the draft mp4 via `VHS_LoadVideoPath`.
- Per-frame face restore via `FaceRestoreCFWithModel` (CodeFormer or GFPGAN).
- 4× upscale via `ImageUpscaleWithModel` + `RealESRGAN_x4plus.pth`.
- Downscale to 1080×1920 via `ImageScale` (lanczos).
- 2× frame interpolation via `RIFE VFI` (16fps → 32fps).
- H.264 encode at CRF 18 via `VHS_VideoCombine`.
- Uploads the refined mp4 to RunPod's network-volume S3 bucket.

## Differences vs `worker-comfyui-wan` (base video gen worker)

Additive only — every node and model the base worker provides is still here.

| Added                         | Source                                   | Purpose                                       |
| ----------------------------- | ---------------------------------------- | --------------------------------------------- |
| `ComfyUI-Frame-Interpolation` | `Fannovel16/ComfyUI-Frame-Interpolation` | `RIFE VFI` node                               |
| `facerestore_cf`              | `mav-rik/facerestore_cf`                 | `FaceRestoreCFWithModel` + model loader       |
| `RealESRGAN_x4plus.pth`       | xinntao Real-ESRGAN releases             | 4× upscale model (base image only had x2plus) |
| `rife47.pth`                  | RIFE 4.7 checkpoint                      | Pre-staged so cold starts skip the download   |

The refinement workflow does not rely on Wan2.2 / FLUX / sage-attention, but those packs are kept in the image because removing them would diverge the Dockerfile too far from the upstream and add maintenance overhead. Image is ~1.5 GB larger than the base; that's the cost of running refinement on a separate endpoint with no risk to base.

## Network volume layout

This worker is intended to attach to the **EUR-IS-1** Wan video volume `t4t7plsew4` so it shares input/output S3 keys with `worker-comfyui-wan`. The refinement workflow reads the draft mp4 from `/runpod-volume/refine-inputs/<uuid>-<filename>.mp4` (uploaded by `VideoStorageManager.upload` from the `@only-scraper/ai-model-video-gen` package) and writes the refined mp4 to `/runpod-volume/video-outputs/<jobId>/refined.mp4`.

## Required env vars (set in RunPod endpoint dashboard)

| Name                                         | Purpose                                  |
| -------------------------------------------- | ---------------------------------------- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Network-volume S3 creds                  |
| `AWS_S3_BUCKET`                              | `t4t7plsew4`                             |
| `AWS_S3_ENDPOINT_URL`                        | `https://s3api-eur-is-1.runpod.io`       |
| `AWS_REGION`                                 | `eur-is-1`                               |
| `S3_OUTPUT_PREFIX`                           | `video-outputs/`                         |

## Build + deploy

The repo is connected to RunPod via the GitHub-driven build flow — pushing to `main` triggers an image rebuild. The endpoint config (GPU pool, idle timeout, workers min/max) is set in the RunPod dashboard, not in this repo.

See `docs/superpowers/plans/2026-05-02-video-refinement-HANDOFF.md` in the `only-scraper` monorepo for the end-to-end deployment runbook.

## Endpoint suggested config

| Setting         | Value                                      |
| --------------- | ------------------------------------------ |
| Region          | `eur-is-1`                                 |
| Network volume  | `t4t7plsew4` mounted at `/runpod-volume`   |
| GPU pool        | A100 80GB (priority) → H100 NVL (fallback) |
| Workers min/max | 1 / 2                                      |
| Idle timeout    | 60s                                        |
| Per-job timeout | 600s (10 min)                              |

## Smoke test

After endpoint deploys, submit `refine-1080p32.json` (from `packages/ai-model-video-gen/src/workflows/` in the `only-scraper` monorepo) via the RunPod dashboard's "Run" panel against an existing test mp4 already on the volume. Expected: 1080×1920 / 32fps mp4 lands in `video-outputs/<jobId>/`.

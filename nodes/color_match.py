import os
import uuid
import shutil
import cv2
import torch
import numpy as np

import folder_paths


class TSColorMatchSequentialBias:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "chunk_size": ("INT", {"default": 81, "min": 1, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "TS Utils/Video"

    EPS = 1e-6

    FPS = 30
    HEAD_WIN = 1
    TAIL_WIN = 1
    SAVE_TEMP_CHUNKS = True  
    DEBUG = False

    @staticmethod
    def _rgb01_to_bgr8(rgb01: np.ndarray) -> np.ndarray:
        out = np.clip(rgb01 * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr8_to_rgb01(bgr8: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    @staticmethod
    def _sample_rgb01_all_pixels_from_bgr8(frame_bgr8: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame_bgr8, (480, 270), interpolation=cv2.INTER_AREA)
        rgb01 = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb01.reshape(-1, 3)

    def _compute_stats(self, frames_bgr8: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pxs = [self._sample_rgb01_all_pixels_from_bgr8(fr) for fr in frames_bgr8]
        px = np.concatenate(pxs, axis=0)
        mean = px.mean(axis=0).astype(np.float32)
        std = px.std(axis=0).astype(np.float32)
        return mean, std

    def _apply_color_transfer_rgb01(
        self,
        frame_bgr8: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        source_mean: np.ndarray,
        source_std: np.ndarray,
    ) -> np.ndarray:
        rgb01 = self._bgr8_to_rgb01(frame_bgr8)
        scale = target_std / (source_std + self.EPS)
        out = (rgb01 - source_mean) * scale + target_mean
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _ensure_mp4v_writer(path: str, fps: int, w: int, h: int) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"Can't open VideoWriter for: {path}")
        return vw

    def process(self, images, chunk_size: int):
        images_np = images.detach().cpu().numpy()
        n = int(images_np.shape[0]) if images_np is not None else 0
        if n <= 0:
            return (images,)

        h = int(images_np.shape[1])
        w = int(images_np.shape[2])

        out_dir = folder_paths.get_output_directory()
        run_id = uuid.uuid4().hex[:10]
        base_name = f"matched_sequential_transfer_{run_id}"

        tmp_dir = os.path.join(out_dir, f".tmp_{base_name}")
        os.makedirs(tmp_dir, exist_ok=True)

        final_path_tmp = os.path.join(tmp_dir, f"{base_name}.mp4")
        final_path = os.path.join(out_dir, f"{base_name}.mp4")

        out_final = None
        out_frames_rgb01: list[np.ndarray] = []

        fps = int(self.FPS)
        head_win = int(self.HEAD_WIN)
        tail_win = int(self.TAIL_WIN)
        save_temp_chunks = bool(self.SAVE_TEMP_CHUNKS)
        debug = bool(self.DEBUG)

        try:
            if debug:
                print("Processing images chunk by chunk and building final video directly...")
                print(f"  N={n}, chunk_size={chunk_size}, fps={fps}, size={w}x{h}")
                print(f"  save_temp_chunks={save_temp_chunks}, head_win={head_win}, tail_win={tail_win}")

            out_final = self._ensure_mp4v_writer(final_path_tmp, fps, w, h)

            ci = 0
            prev_modified_tail_bgr8: list[np.ndarray] = []

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                length = end - start

                raw_frames_bgr8: list[np.ndarray] = []
                for i in range(start, end):
                    raw_rgb01 = images_np[i].astype(np.float32)
                    raw_frames_bgr8.append(self._rgb01_to_bgr8(raw_rgb01))

                processed_frames_bgr8: list[np.ndarray] = []

                if ci == 0:
                    processed_frames_bgr8 = raw_frames_bgr8
                    if debug:
                        print(f"Chunk {ci}: Keep original (base for next)")
                else:
                    target_m, target_s = self._compute_stats(prev_modified_tail_bgr8)

                    h1 = min(head_win, length)
                    raw_head_frames_bgr8 = raw_frames_bgr8[:h1]
                    source_m, source_s = self._compute_stats(raw_head_frames_bgr8)

                    if debug:
                        print(f"Chunk {ci-1}->{ci}:")
                        print(f"  Target Mean: {target_m}, Std: {target_s}")
                        print(f"  Source Mean: {source_m}, Std: {source_s}")

                    for fr_bgr8 in raw_frames_bgr8:
                        rgb01 = self._apply_color_transfer_rgb01(fr_bgr8, target_m, target_s, source_m, source_s)
                        res_bgr8 = self._rgb01_to_bgr8(rgb01)
                        processed_frames_bgr8.append(res_bgr8)

                for fr_bgr8 in processed_frames_bgr8:
                    out_final.write(fr_bgr8)
                    out_frames_rgb01.append(self._bgr8_to_rgb01(fr_bgr8))

                tw = min(tail_win, length)
                tail_start = max(0, length - tw)
                prev_modified_tail_bgr8 = processed_frames_bgr8[tail_start:]

                if save_temp_chunks:
                    chunk_filename_tmp = os.path.join(tmp_dir, f"chunk_{ci:03d}.mp4")
                    out_chunk = self._ensure_mp4v_writer(chunk_filename_tmp, fps, w, h)
                    for fr_bgr8 in processed_frames_bgr8:
                        out_chunk.write(fr_bgr8)
                    out_chunk.release()
                    if debug:
                        print(f"Saved TEMP chunk_{ci:03d}.mp4 with {length} frames.")

                ci += 1

            out_final.release()
            out_final = None

            shutil.move(final_path_tmp, final_path)

            if debug:
                print("\nDONE. Saved final video:", final_path)

        finally:
            if out_final is not None:
                try:
                    out_final.release()
                except Exception:
                    pass

            try:
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

        out_tensor = torch.from_numpy(np.stack(out_frames_rgb01, axis=0).astype(np.float32))
        return (out_tensor,)

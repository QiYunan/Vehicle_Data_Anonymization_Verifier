import os
import subprocess
import json
import traceback

def get_vedio_info(ffmpeg_path, video_path):
    ffprobe_path = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmoeg","ffprobe")
    cmd = [ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'json', video_path]
    try:
        result = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, check = True)
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            streams = info['streams'][0]
            width = int(stream.get('width',0))
            height = int(stream.get('height',0))
            duration = float(stream.get('duration',0))
            fs_parts = stream.get('r_frame_rate', '30/1').split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            return width, height, duration
    except Exception:
        return None
    return None

def process_single_video(ffmpeg_path, video_path, root_output_dir):
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(root_output_dir, video_filename)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    meta = get_video_info(ffmpeg_path, video_path)
    if meta:
        w, h, fps, duration = meta
        print(f"\n[视频]:{os.path.basename(video_path)}")
        print(f"分辨率: {w}x{h} | 帧率: {fps:.2f} | 时长：{duration:.2f}秒")
        
    output_pattern = os.path.join(video_output_dir,"gbt_frame_%04d.jpg")
    
    cmd 
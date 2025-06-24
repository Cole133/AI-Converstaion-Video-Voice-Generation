import tkinter as tk
from tkinter import simpledialog
import random
import openai
import requests
import os
from dotenv import load_dotenv
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip,
    ImageClip, concatenate_audioclips, TextClip, ColorClip
)
from PIL import Image

import moviepy.config as mpy_config
mpy_config.change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})



# --- CONFIG ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
SONIC_VOICE_ID = os.getenv("SONIC_VOICE_ID")
TAILS_VOICE_ID = os.getenv("TAILS_VOICE_ID")
CHARACTERS = [
    {"name": "Sonic", "voice_id": SONIC_VOICE_ID, "image": r"C:\ClipTok\Char\sonic.png", "side": "left"},
    {"name": "Tails", "voice_id": TAILS_VOICE_ID, "image": r"C:\ClipTok\Char\tails.png", "side": "right"},
]
GAMEPLAY_DIR = r"C:\ClipTok\Gameplay"
OUTPUT_VIDEO = "output_clip.mp4"
VIDEO_RES = (1080, 1920)  # TikTok vertical

def get_topic():
    """Prompt user for a discussion topic."""
    root = tk.Tk()
    root.withdraw()
    topic = simpledialog.askstring("AI Clip Automation", "Enter a topic for Sonic & Tails to discuss:")
    root.destroy()
    return topic

def generate_dialog(topic, n_turns=8):
    """Generate dialog using OpenAI."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        f"Write a fun, back-and-forth dialog script between Sonic and Tails about '{topic}'. "
        "Alternate lines, keep it conversational, and make it about 60-90 seconds long. "
        "Format as:\nSonic: ...\nTails: ...\nSonic: ...\nTails: ...\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=1.0,
    )
    script = response.choices[0].message.content
    dialog = []
    for line in script.splitlines():
        if ":" in line:
            char, text = line.split(":", 1)
            char = char.strip()
            text = text.strip()
            if char.lower() in ["sonic", "tails"]:
                dialog.append((char, text))
    return dialog

def tts_elevenlabs(text, voice_id, out_path):
    """Generate TTS audio using ElevenLabs."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}
    }
    r = requests.post(url, headers=headers, json=data)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
    else:
        raise Exception(f"ElevenLabs TTS failed: {r.text}")

def get_gameplay_clips(target_duration):
    """Select and trim gameplay video(s) to match target duration."""
    files = [os.path.join(GAMEPLAY_DIR, f) for f in os.listdir(GAMEPLAY_DIR) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))]
    if not files:
        print("No gameplay files found!")
        return None
    random.shuffle(files)
    clips = []
    total = 0
    i = 0
    while total < target_duration and files:
        clip = VideoFileClip(files[i % len(files)])
        clips.append(clip)
        total += clip.duration
        i += 1
        if len(files) == 1:
            break
    if not clips:
        print("No video clips loaded!")
        return None
    if len(clips) == 1 and clips[0].duration > target_duration:
        final = clips[0].subclip(0, target_duration)
    else:
        final = concatenate_videoclips(clips)
        if final.duration > target_duration:
            final = final.subclip(0, target_duration)
    return final

def make_character_overlay(image_path, side, video_size, height=800):
    """Create a character overlay image, positioned off-center."""
    img = Image.open(image_path)
    aspect = img.width / img.height
    new_height = height
    new_width = int(aspect * new_height)
    img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    img.save("_tmp_char.png")
    # Off-center: Sonic left, Tails right
    if side == "left":
        x = int(video_size[0] * 0.08)  # 8% from left
    else:
        x = int(video_size[0] * 0.60)  # 60% from left
    y = video_size[1] - new_height - 50
    return ImageClip("_tmp_char.png").set_position((x, y)).set_duration(0.1)

def transcribe_audio_to_captions(audio_path):
    """Transcribe audio to caption segments using Whisper."""
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['segments']

def make_word_group_caption(words, highlight_idx, start, end, x, y):
    """Create centered caption with one highlighted word layered on top."""
    line_text = " ".join(words)
    highlight_word = words[highlight_idx]

    base_clip = TextClip(
        line_text,
        fontsize=70,
        font='Arial',
        color='white',
        method='caption',
        size=(VIDEO_RES[0] * 0.8, None),
        align='center'
    ).set_position(('center', y)).set_start(start).set_end(end)

    # Calculate x-offset for highlighted word (approximate by measuring prefix width)
    prefix = " ".join(words[:highlight_idx])
    pre_clip = TextClip(
        prefix,
        fontsize=70,
        font='Arial',
        method='caption'
    )
    pre_width = pre_clip.w if prefix else 0

    highlight_clip = TextClip(
        highlight_word,
        fontsize=70,
        font='Arial',
        color='lime',
        method='caption'
    ).set_position((
        ('center', y),
    )).set_start(start).set_end(end)

    return CompositeVideoClip([base_clip, highlight_clip])

def main():
    topic = get_topic()
    if not topic:
        print("No topic entered.")
        return

    dialog = generate_dialog(topic)
    print("Generated dialog:")
    for char, line in dialog:
        print(f"{char}: {line}")

    # Generate TTS for each line and build a timeline
    audio_segments = []
    char_timeline = []
    current_time = 0
    for i, (char, line) in enumerate(dialog):
        char_info = next(c for c in CHARACTERS if c["name"].lower() == char.lower())
        audio_path = f"line_{i}_{char}.mp3"
        tts_elevenlabs(line, char_info["voice_id"], audio_path)
        audio_clip = AudioFileClip(audio_path)
        audio_segments.append(audio_clip)
        char_timeline.append((char_info, current_time, current_time + audio_clip.duration))
        current_time += audio_clip.duration

    if not audio_segments:
        print("No audio was generated. Exiting.")
        return

    # Concatenate all audio
    full_audio = concatenate_audioclips(audio_segments)
    total_duration = full_audio.duration

    # Export full audio for Whisper
    full_audio.write_audiofile("full_audio.wav")

    # Get gameplay video
    gameplay = get_gameplay_clips(total_duration)
    if gameplay is None:
        print("No gameplay video found. Exiting.")
        return
    gameplay = gameplay.resize(VIDEO_RES)

    # Build character popups timeline
    overlays = []
    for char_info, start, end in char_timeline:
        overlay = make_character_overlay(char_info["image"], char_info["side"], VIDEO_RES)
        duration = end - start
        slide_duration = min(0.5, duration / 3)
        img_width = overlay.w

        # Sonic slides from off left; Tails slides from off right
        if char_info["side"] == "left":
            x_in = int(VIDEO_RES[0] * 0.08)
            x_out = -img_width
        else:
            x_in = int(VIDEO_RES[0] * 0.60)
            x_out = VIDEO_RES[0] + img_width

        y = VIDEO_RES[1] - overlay.h - 50

        def pos_func(t, x_in=x_in, x_out=x_out, y=y, slide_duration=slide_duration, duration=duration):
            if t < slide_duration:
                progress = t / slide_duration
                x = x_out + (x_in - x_out) * progress
            elif t > duration - slide_duration:
                progress = (t - (duration - slide_duration)) / slide_duration
                x = x_in + (x_out - x_in) * progress
            else:
                x = x_in
            return (x, y)

        overlays.append(
            overlay.set_start(start)
                .set_end(end)
                .set_position(lambda t: pos_func(t))
        )

    # --- Whisper captions ---
    print("Transcribing audio with Whisper...")
    segments = transcribe_audio_to_captions("full_audio.wav")
    print("Whisper transcription complete.")

    captions = []
    group_size = 4  # 3-5 words per group
    y_offset = VIDEO_RES[1] - 300

    for seg in segments:
        words = seg['text'].split()
        word_count = len(words)
        word_duration = (seg['end'] - seg['start']) / max(1, word_count)
        for i in range(word_count):
            group_start = max(0, i - group_size//2)
            group_end = min(word_count, group_start + group_size)
            group_words = words[group_start:group_end]
            highlight_idx = i - group_start
            w_start = seg['start'] + i * word_duration
            w_end = w_start + word_duration

            # --- Center captions horizontally ---
            char_x = 'center'
            y = y_offset

            captions.append(make_word_group_caption(
                group_words, highlight_idx, w_start, w_end, char_x, y
            ))

    # Compose final video
    final = CompositeVideoClip([gameplay] + overlays + captions)
    final = final.set_audio(full_audio)

    final.write_videofile(
        OUTPUT_VIDEO,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=6,
        preset='ultrafast',
        bitrate="2M"   # 2 Mbps, adjust as needed
    )


    # Cleanup
    for i in range(len(dialog)):
        try:
            os.remove(f"line_{i}_{dialog[i][0]}.mp3")
        except Exception:
            pass
    if os.path.exists("_tmp_char.png"):
        os.remove("_tmp_char.png")
    if os.path.exists("full_audio.wav"):
        os.remove("full_audio.wav")

    print(f"Done! Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
# Prompt Examples for JoyAI-Video-Edit

Use text prompts to guide video editing. Prompts work with both `run.sh` (image + prompt) and `run_text.sh` (text only).

## Quick Start

```bash
# Text-only mode
./run_text.sh assets/input.mp4 outputs/out.mp4 "watercolor painting" 10 auto auto 20 7.5

# Debug with prompt
./run_debug.sh --prompt "anime art style"
```

## Prompt Categories

### 🎨 Style & Artistic

- "watercolor painting with soft edges"
- "oil painting with thick visible brushstrokes"
- "anime art style"
- "cartoon illustration style"
- "photorealistic cinematic"
- "van gogh style impressionist"
- "pixel art style"
- "pencil sketch style"
- "stained glass art"
- "line art drawing"

### 🌅 Lighting & Atmosphere

- "golden hour sunset lighting"
- "soft warm ambient lighting"
- "dark moody low-key lighting"
- "bright daylight overexposed"
- "neon cyberpunk lighting"
- "foggy misty atmosphere"
- "underwater lighting"
- "candlelight"
- "moonlit night"
- "studio lighting"

### 🎬 Color & Mood

- "warm orange and red tones"
- "cool blue and purple tones"
- "black and white grayscale"
- "sepia toned vintage"
- "high saturation vibrant colors"
- "desaturated muted colors"
- "monochrome blue"
- "retro 70s color palette"

### 🌍 Environment & Scene

- "snowing heavily"
- "raining with wet surfaces"
- "golden sunset sky"
- "stormy dark clouds"
- "underwater scene"
- "forest setting"
- "urban city environment"
- "desert sandy landscape"
- "beach tropical setting"
- "space with stars"

### 👤 Character & Object Edits

- "person wearing red jacket"
- "person with long flowing hair"
- "make everyone smile"
- "add a hat to the person"
- "make the character look older"
- "make the character look younger"
- "add glasses to everyone"
- "change to formal dress"

### 🔧 Technical Edits

- "increase sharpness and detail"
- "soft focus blurred background"
- "depth of field focus on foreground"
- "motion blur effect"
- "lens flare effect"
- "chromatic aberration"
- "film grain noise"

### 🎯 Complex Prompts

Combine concepts for stronger results:

- "watercolor painting style with warm golden sunset lighting"
- "anime art style with neon cyberpunk lighting and dark moody atmosphere"
- "photorealistic with soft warm studio lighting and high saturation colors"
- "oil painting impressionist style with foggy misty morning atmosphere"
- "monochrome black and white with high contrast dramatic lighting"

## Tips for Better Results

1. **Be specific** — "watercolor" works better than "painting"
2. **Combine concepts** — "watercolor + golden sunset" = stronger effect
3. **Include lighting** — "watercolor with soft warm lighting" = more cohesive
4. **Use CFG scale** — Higher CFG (7.5-15) for stronger guidance, lower (3-5) for subtlety
5. **Increase steps** — More diffusion steps (20-50) for better quality
6. **Simple prompts first** — Test with "watercolor" before "oil painting with thick brushstrokes"

## Examples by Use Case

### Quick Test
```bash
./run_text.sh assets/input.mp4 outputs/test.mp4 "watercolor" 5 auto auto 10 7.5
```

### High Quality
```bash
./run_text.sh assets/input.mp4 outputs/hq.mp4 "watercolor painting with golden sunset" 10 auto auto 50 7.5
```

### Subtle Effect
```bash
./run_text.sh assets/input.mp4 outputs/subtle.mp4 "slightly desaturated warm tones" 10 auto auto 20 3.0
```

### Debug & Inspect
```bash
./run_debug.sh --prompt "watercolor painting style"
```

## Prompt Performance

Tested prompts (best results):
- ✅ "watercolor painting" — Consistent, smooth style transfer
- ✅ "anime art style" — Strong stylization
- ✅ "golden sunset lighting" — Atmospheric changes
- ✅ "photorealistic cinematic" — Detail enhancement
- ✅ "oil painting impressionist" — Artistic transformation

Challenging prompts:
- ⚠️ Very specific characters — May not preserve identity
- ⚠️ Extreme color changes — May need higher CFG
- ⚠️ Multiple simultaneous edits — Focus on one concept at a time

## Troubleshooting

**Prompt has no effect:**
- Increase `--cfg` (try 10-15)
- Increase `--steps` (try 30-50)
- Use simpler prompt first

**Output looks distorted:**
- Lower `--cfg` (try 3-5)
- Use fewer steps initially
- Check with `./run_debug.sh --prompt "your_prompt"`

**Takes too long:**
- Reduce `--frames`
- Reduce `--steps` to 10-15
- Use smaller resolution

**Output is pixelated:**
- Check intermediate frames: `./run_debug.sh --prompt "watercolor"`
- Try text-only mode: `./run_text.sh`
- Increase steps for better convergence

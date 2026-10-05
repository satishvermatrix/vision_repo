# Video sources & licenses

| file | source | license | notes |
|---|---|---|---|
| `sintel_trailer.mp4` | https://media.w3.org/2010/05/sintel/trailer.mp4 | CC-BY 3.0 - Blender Foundation (durian project) | 854x480, 52 s |
| `sintel_10s.mp4`, `sintel_30s.mp4` | cut from `sintel_trailer.mp4` with ffmpeg (no audio) | same | duration-cut variants |
| `bbb_1080.mp4` | https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_5MB.mp4 | CC-BY 3.0 - Blender Foundation (peach project), encoded by test-videos.co.uk | 1920x1080, 10 s |
| `bbb_720.mp4`, `bbb_360.mp4` | transcoded from `bbb_1080.mp4` with ffmpeg (same content, lower resolution) | same | resolution ladder |
| `jellyfish_720.mp4` | https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_5MB.mp4 | free test footage, test-videos.co.uk | 1280x720, 10 s, real footage |
| `flower.mp4` | https://mdn.github.io/shared-assets/videos/flower.mp4 | CC0 / public domain (MDN shared assets) | 960x540, 5 s, real footage |
| `data/frames/*` | extracted frames from the above (ffmpeg) | same | used in frame-list mode experiments |

All video variants were produced locally with ffmpeg: `-an -c:v libx264 -crf 23 -pix_fmt yuv420p` (+ `scale` / `-t` filters).

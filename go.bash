# uv run render.py --command render --fps 60 --num_objects 2000 --out "apple.mp4"  --halo
uv run render.py --command render --fps 60 --num_objects 2000 --img "ppt/loong2.png" --out "loong.mp4"  --halo
uv run render.py --command render --fps 60 --num_objects 3000 --out "speit.mp4"  --halo --img "ppt/speit_sq.png" --offset_ood_prop 0.5  --halo_intensity 0.015 --ax_offset 0.0
uv run render.py --command render --fps 60 --num_objects 5000 --out "sjtu.mp4"   --halo --img "ppt/sjtu.png"     --offset_ood_prop 0.35 --halo_intensity 0.01 --img_scale 1.2 --ax_offset 0.0


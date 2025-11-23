uv run render.py --command render --fps 60 --num_objects 1000 --deceleration_rate 5.0 --img "ppt/loong2.png" --out "loong.mp4" --offset_ood_prop 0.6 --halo
uv run render.py --command render --fps 60 --num_objects 3000 --deceleration_rate 5.0 --img "ppt/speit.png" --out "speit.mp4" --offset_ood_prop 0.2 --halo --halo_intensity 0.005
uv run render.py --command render --fps 60 --num_objects 5000 --deceleration_rate 5.0 --img "ppt/sjtu.png" --out "sjtu.mp4" --offset_ood_prop 0.12 --halo --halo_intensity 0.003

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os


task_name = "push"
success_run = "1"
image_path = os.path.join(task_name, task_name+"_success_"+success_run)
sample_image_path = image_path + "/plane_view_episode_0_timestep_0.svg"
drawing = svg2rlg(sample_image_path)
renderPM.drawToFile(drawing, image_path +
                    "/plane_view_episode_0_timestep_0.png", fmt='PNG')

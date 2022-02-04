import subprocess
import os

object_count = 0

for x in os.listdir("input_images"):
    input_image = "input_images/" + x
    object_count += 1
    subprocess.call(['ffmpeg', '-i', input_image, '-filter:v', 'crop=182:415:545:301', f'output_images/object_{object_count}.png'])
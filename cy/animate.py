"""
Convert image sequence to video

Usage:
    animate.py <sim_name> [--fps=<fp>]

Options:
    --fps=<fp>  Framerate [default: 24]
"""

from docopt import docopt
import pathlib
import glob
from natsort import natsorted
import moviepy.video.io.ImageSequenceClip

args = docopt(__doc__)
sim_name = args['<sim_name>']
fps = int(args['--fps'])
img_dir = pathlib.Path('frames').absolute()
img_dir = img_dir.joinpath(sim_name)

files = glob.glob(str(img_dir.joinpath("*.jpg")))
image_files = natsorted(files)

movie_path = pathlib.Path('movies').absolute()
output_path = movie_path.joinpath(sim_name+".mp4")

if not movie_path.exists():
    movie_path.mkdir()

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(str(output_path))
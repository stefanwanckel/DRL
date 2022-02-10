import imageio

from os import listdir
from os.path import isfile, join

task_name = 'pick_and_place'
test_run_name = 'test_2_success'

folder = 'EXPERIMENTS'
#folder = 'STUDY_20'

mypath = join(task_name, test_run_name)
onlyfiles = [f for f in listdir(mypath) if (f.endswith('png'))]

onlyfiles.sort()

images = []
for filename in onlyfiles:
    images.append(imageio.imread(join(mypath, filename)))

kargs = {'duration': 0.2}
imageio.mimsave(mypath + '/images.gif', images, **kargs)
imageio.mimsave(mypath + '/images.mp4', images)

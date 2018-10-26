from setuptools import setup

with open('requirements.txt') as fle:
    dependencies = fle.readlines()

setup(name='Chaotic Particle Swarm',
      version='0.1',
      description='Improved particle swarm optimization combined with chaos. doi:10.1016/j.chaos.2004.11.095',
      author='Jim Lemmers',
      author_email='shout@jimlemmers.com',
      licenses='BSD',
      packages=['chaotic_particle_swarm'],
      install_requires=dependencies,
      zip_safe=False)

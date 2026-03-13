MAGPIE - MAGnetosheath Particle Interaction Explorer

To import the functions as it stands, at the top of the notebook do:

import requests

base_url = 'https://raw.githubusercontent.com/Parker20019027/magpie/main/'
files = ['boxcar_averager.py', 'current_density.py', 'field_particle_correlation.py', 'pressure_strain.py']

for f in files:
  response = requests.get(base_url + f)
  with open(f, 'w', encoding='utf-8') as file:
    file.write(response.text)

from boxcar_averager import boxcar_averager
from current_density import current_density
from field_particle_correlation import field_particle_correlation
from pressure_strain import pressure_strain

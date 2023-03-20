from setuptools import setup, find_packages


setup(name='NCPLR',
      version='1.0.0',
      description='Neighbour Consistency Guided Pseudo-Label Refinement for Unsupervised Person Re-Identification',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
          'Label Refinement'
      ])

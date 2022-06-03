FROM dolfinx/lab:v0.4.1

RUN pip install 'cython>=0.*,<1.*' && \
    pip install 'pygmsh>=7.*,<8.*' 'meshio>=5.*,<6.*' 'pyacvd>=0.*,<1.*' 'tetgen>=0.*,<1.*' \
                'panel>=0.*,<1.*' 'vtk>=9.*,<10.*' 'numpyencoder>=0.*,<1.*' && \
    pip install git+https://github.com/MABilton/tetridiv

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

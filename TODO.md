# TODO

- [ ] cambiar returns multiples a diccionarios (así cuando se olvida en que
      posición del return está cada valor, puede encontrarselo con las keys
      natoms, box_size, atom_type, etc).

- [ ] cambiar API de manera tal que no quede rdf.gofr si no solo rdf (por ejemplo).
      Esto hacerlo cambiando el lugar en el que se encuentran los .py pero dejando
      los modulos de C en carpetas separadas (puede ser una sola `src/`). Tener
      cuidado con el setup.py

- [ ] triclinic (non-orthogonal) boxes for the different modules.

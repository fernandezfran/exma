# TODO

- [ ] cambiar returns multiples a diccionarios (así cuando se olvida en que
      posición del return está cada valor, puede encontrarselo con las keys
      natoms, box_size, atom_type, etc).

- [ ] cambiar API de manera tal que no quede rdf.gofr si no solo rdf (por ejemplo).
      Esto hacerlo cambiando el lugar en el que se encuentran los .py pero dejando
      los modulos de C en carpetas separadas (puede ser una sola `src/`). Tener
      cuidado con el setup.py

- [ ] juntar monoatomic y diatomic en una sola función (total cuando se la llama
      se le pasa que ambos son el mismo tipo de átomo, como en VMD).

- [ ] agregar opción de cluster a los distintos analisis, como en rdf.

- [ ] agregar _ al final de las variables que son dependientes en el __init__, por
      ejemplo N_a_ es el nro de particulas de tipo A. (Viene de sklearn)

- [ ] triclinic (non-orthogonal) boxes for the different modules.

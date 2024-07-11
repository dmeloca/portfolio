---
title: "El cero detrás de las matemáticas"
date: 2024-04-14
layout: post
---
# El cero detrás de las matemáticas
En los axiomas de Peano, usados para la construcción de los números naturales y además para las pruebas de inducción matemática, originaron unas de las discusiones más intrigantes de la matemática: ¿Es el cero un elemento de los naturales? 

En "Arithmetices principia" Peano (1889), plantea en los axiomas que el 1 es el elemento especial y no el cero como se plantea actualmente. Entonces ¿Cuál es la razón de incluir el cero en los naturales según la teoría de conjuntos y la teoría de grupos?

En este escrito se pretende abordar de manera superficial las respuestas a las preguntas anteriormente planteadas. Para lo cual, se describirá los planteamientos desde la teoría de conjutos y la teoría de grupos. 
#### 1. Teoría de conjuntos
En primer lugar bajo el intento de formalizar la aritmética, según Fernández y Miranda (2016) [**R. Dedekind**](https://virtual.uptc.edu.co/ova/estadistica/docs/autores/pag/mat/Dedekind-1.asp.htm) quien en su libro "¿Qué son y para que sirven los números?" define a los naturales de foma conjuntista y cuyo primer elemento es 1. Dedekind construye su definición a partir de las comunidades de cadenas de las cuales A es parte y por medio de la construcción se refiere a que el número inicial es 1.

Posteriormente, como lo mencionan Fernández y Miranda (2016), [**Ernst Zermelo**](https://matematics.wordpress.com/2019/11/13/ernst-friedrich-ferdinand-zermelo/) y [**Adolf Fraenkel**](https://mathshistory.st-andrews.ac.uk/Biographies/Fraenkel/) postulan el sistema axiomatico mejor conocido como ZF por el cual se puede hacer otra construcción de este sistema numérico. La cual consiste en empezar desde un conjunto vacío el cual se identifica con el símbolo 0, que mediante ZF se puede determinar que ese conjuto vacío es único, y como función sucesor \\(x\bigcup \\{x\\}\\), esta forma de representar los números naturales permite contar mediante una biyección entre un natural y el conjunto que se quiere contar, definición dada por [**Bertrand Russell**](https://www.filosofos.net/russell/russell_bio.htm).

Lo que demuestra, que el cero se incluye para realizar la construcción de los naturales a partir del sistema axiomático ZF
#### 2. Teoría de grupos
Al definir una estructura algebraíca, tomando el conjunto dado por los axiomas de Peano y una operación binaria \\(+:\mathbb{N}\times\mathbb{N}\to\mathbb{N}: (a,b)\to a+b\\), esta cumple con la propiedad de asociatividad. Luego, en este punto hay una distinción, pues si se toma el cero como elemento de los naturales, la operación cuenta con elemento neutro; sin embargo, si se toma como el elemento inicial al 1 la operación no cumple la propiedad modulativa. En ambos casos vemos que con el cero o sin él, no se logra que esta estructura algebráica sea un grupo.

Por lo cual, se evidencia que la inclusión del cero se realiza para agregar una propiedad modulativa a la operción binaria.

Para concluir, se aprecia que el cero es requerido en ambas teorías con fines particulares, pero no hace una gran diferencia en los campos ya mencionados. Entonces, queda a elección del matemático o persona afín a esta rama que decida, dependiendo de sus necesidades y creencias, si incluye al 0 como un elemento de los naturales o no.

### Referencias:
- Fernández, T y Miranda, C. (2016). "Los Números Naturales desde la Teoría de
Conjuntos y la Teoría de Categorías". Trabajo de grado para optar por el título de Licenciados en matemáticas y física, Universidad del Valle. https://bibliotecadigital.univalle.edu.co/server/api/core/bitstreams/1cc4b6af-5484-4e09-a8b8-f3c6aa1df5f2/content
- Peano, G. (1889). "Arithmetices principia".  Fratres Bocca, Roma.



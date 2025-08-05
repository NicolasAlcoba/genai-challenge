# Experimentos y Mejoras Realizadas

## Lo que probé y por qué

Acá documento todo lo que fui probando para mejorar el sistema RAG y las conclusiones a las que llegué.

## 1. Métodos de RAG

### Empecé probando diferentes métodos de RAG
Al principio pensé que el problema podía estar en cómo estaba haciendo el retrieval, así que probé:
- RAG básico (el que venía de base)
- RAG híbrido combinando búsqueda semántica con keywords
- RAG balanceado con weights en los resultados

**Resultado**: No mejoró mucho la cosa. Las diferencias eran mínimas, así que decidí buscar el problema en otro lado.

## 2. Métodos de Chunking

### Cambié el enfoque: cómo estaba guardando los datos
Como los métodos de RAG no dieron resultado, me puse a experimentar con diferentes formas de dividir y guardar la info:

- **Simple Chunking**: División fija por tamaño (el básico)
- **Sentence Chunking**: Dividir por oraciones completas
- **Paragraph Chunking**: Dividir por párrafos
- **Sliding Window**: Ventanas que se superponen un poco
- **Semantic Chunking**: División basada en similitud semántica
- **Structural Chunking**: Respetando la estructura del documento
- **Hybrid Chunking**: Mezclando varios métodos

## 3. Revisé las preguntas de evaluación

Me puse a revisar si las preguntas que estaba usando para evaluar estaban bien hechas. Las miré una por una y me parecieron correctas, así que el problema no era ese.

### Lo que me quedó pendiente
Me di cuenta de que tendría que haber analizado más en detalle:
- Cuántos datos le estoy pasando al modelo con cada método
- Qué tan bueno es el contexto que recupero
- Si el tamaño de los chunks tiene que ver con la calidad

No llegué a profundizar en esto porque se me fue mucho tiempo en los experimentos anteriores, pero creo que ahí está la clave. Todos los métodos que probé pueden funcionar si los ajusto bien.

## 4. Experimento con prompts

### Intenté usar prompts más complejos
Como tengo experiencia con modelos mas grandes pensé, "bueno, voy a hacer un prompt más detallado como hago siempre".

Con modelos chicos esto fue contraproducente. Los prompts largos y detallados confundieron al modelo en vez de ayudarlo. Las respuestas empeoraron porque los modelos pequeños prefieren instrucciones simples y directas, por eso seguí enfocándome en mejorar el RAG y el chunking en vez de complicar las instrucciones.

Tambien agregue accelerate para poder cargarlo a mi gpu, pero no lo conte como mejora

Aclarar que para esto estuve apoyado tanto en internet y LLMs (mas que nada para escribir)
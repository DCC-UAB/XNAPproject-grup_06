## **FASE 1**

**Pregunta 1:** Està desbalancejat el nostre dataset? <br />
No és rellavant aquest aspecte.
<br /> <br />

**Pregunta 2:** Podria ser beneficiós l'aprenentage de cadascuna de les lletres per separat? Podria servir per mirar el balanceig del dataset? <br />
En la fase següent, es realitzaran proves de contrast comprovant l'aprenentatge per separat i l'aprenentatge conjunt.
<br /><br />

**Pregunta 3:** Ens resulta útil aplicar data augmentation per aquest projecte? <br />
Sí, generalment, sempre resulta útil.
<br /><br />

**Pregunta 4:** Quins són els millors mètodes per netejar i normalitzar les dades per garantir un rendiment òptim dels nostres models CNN i RNN? <br />
Les dades es troben normalitzades i estandaritzades, ja que son imatges. Únicament, s'han de llegir (binari). Respecte a la part de la neteja, es deixaràn de tenir en compte imatges "il·legibles", com ara paraules tallades per la meitat.
<br /><br />

**Pregunta 5:** Quina ha de ser l'arquitectura dels nostres models CNN i RNN? <br />
Es definirà posteriorment, en haver realitzat diverses proves de contrast.
<br /><br />

**Pregunta 6:** Quantes capes ha d tenir la nostra xarxa?<br />
Es definirà posteriorment, en haver realitzat diverses proves de contrast.
<br /><br />

## **FASE 2**

**Pregunta 1:** Quins models podem fer servir? <br />
CNN, RNN i CRNN.
<br /><br />

**Pregunta 2:** Perquè es poden fer servir els models anteriors?<br />
CNN permet detectar carcaterístiques de les imatges. RNN permet capturar dependències seqüencials a llarg termini, que serveix pel reconeixement de paraules. CRNN combina les característiques de les dues xarxes anteriors.
<br /><br />

**Pregunta 3:** Es pot fer servir una MLP complexa?<br />
No és la opció més òptima, per la manca de capacitat per capturar relacions espacials i "poca escalabilitat", entre vàries.
<br /><br />

**Pregunta 4:** És més convenient un model on l'aprenentage de cadascuna de les lletres es realitza per separat o de forma conjunta?<br />
**Resposta 4:** 
<br /><br />

**Pregunta 5:** És més convenient un model on l'aprenentage de cadascuna de les lletres es realitza per separat o de forma conjunta?<br />
Creiem que el més convenient és CRNN, ja que és un model híbrid entre CNN i RNN que combina la capacitat de les CNNs per extreure característiques visuals i la capacitat de les RNNs per modelar seqüències. 
<br /><br />

**Pregunta 6:** De tots els models, quin és el més convenient? Perquè?<br />
Son més complexes de dissenyar i entrenar que les CNNs o RNNs, i requereixen més recursos computacionals.
<br /><br />

**Pregunta 7:** Quins son els paràmetres del model més convenients? Perquè?<br />
**Resposta 7:** 
<br /><br />

**Pregunta 8:** Quina mètrica és la més òptima per avaluar el rendiment del model? Perquè?<br />
Es poden fer servir les mètriques: Accuracy, Precision, Recall, F1-score?, l'error de reconeixement de caràcters (Character Recognition Error), Word Error Rate (WER).
<br /><br />


<br /><br />

> [!NOTE]
> Com a convenient s'entèn que fa que el model aprengui millor (== més accuracy).

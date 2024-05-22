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
<br /><br /><br />

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
Per a reconèixer noms escrits a mà, és més convenient un model que aprengui de forma conjunta, és a dir, que reconegui paraules completes. Això permet aprofitar el context de les lletres dins de les paraules, la qual cosa sol conduir a una major precisió en les prediccions. Tot i això, en el cas de tenir un conjunt de dades límitat podria ser una bona opció entrenar un model per reconèixer lletres individuals.
<br /><br />

**Pregunta 5:** Quin és el model més convenient?<br />
Creiem que el més convenient és CRNN, ja que és un model híbrid entre CNN i RNN que combina la capacitat de les CNNs per extreure característiques visuals i la capacitat de les RNNs per modelar seqüències. 
<br /><br />

**Pregunta 6:** Quins son els desavantatges del model més convenient?<br />
Son més complexes de dissenyar i entrenar que les CNNs o RNNs, i requereixen més recursos computacionals.
<br /><br />

**Pregunta 7:** Quins son els paràmetres del model més convenients? Perquè?<br />
**Resposta 7:** 
<br /><br />

**Pregunta 8:** Quina mètrica és la més òptima per avaluar el rendiment del model?<br />
Es poden fer servir les mètriques: Accuracy, Precision, Recall, F1-score?, l'error de reconeixement de caràcters (Character Recognition Error), Word Error Rate (WER).
<br /><br />

**Pregunta 9:** Què aporten cadascuna de les mètriques anteriors?<br />
- Accuracy: Nombre de prediccions correctes realitzades respecte a la proproció total de prediccions fetes. Serveix per avaluar, de forma general, com aprèn el model.
- Precision: Nombre de prediccions positives realitzades correctament respecte el nombre total de casos positius. Serveix per avaluar la capacitat del model per identificar correctament els casos positius.
- Recall: Proporció de tots els casos positius reals que el model ha identificat correctament. Serveix per avaluar la capacitat del model per identificar correctament els casos positius reals.
- F1-score: Combina les dues mètriques anteriors, Precision i Recall. Serveix per avaluaar tant la capacitat del model per identificar correctament les instàncies positives com la seva capacitat per evitar falsos positius.
- L'error de reconeixement de caràcters (Character Recognition Error)
- Taxa d'error de les paraules (Word Error Rate)
<br /><br />

**Pregunta 10:** Quantes capes ha d tenir la nostra xarxa?<br />
<br /><br />

**Pregunta 11:** Canvia molt si s'executa amb un sample més petit l'aprenentatge del model CRNN-in-keras-v1?<br />
Si, canvia de forma dràstica. Es pot observar que el rediment del model decreix de forma notable. 
Per tal de contestar aquesta pregunta s’ha canviat el volum de les dades utilitzades reduint-les un 90% i s’ha realitzat la meitat de epochs (perquè el cost computacional sigui assequible en un entorn local) a l'hora de entrenar el model. Amb aquests canvis, els valors d’execució: 
- Correct characters predicted : 87.26%
- Correct words predicted      : 74.53%
<br /><br />
canvien a els valors: 
- Correct characters predicted : 7.47%
- Correct words predicted      : 0.00%
<br /><br />
Amb els valors obtinguts es pot confirmar que l'aprenentatge canvia negativament, de forma que el model no apren de manera correcta. Per tant, es pot afirmar que s'ha d'utilitzar un gran volum de dades i un nombre d'epochs elevat per tal d'obtenir uns bons resultats pel nostre model.
<br /><br />

**Pregunta 12:** Seria útil realitzar un HeatMap?<br />
Creiem que si, perquè pot ajudar a visualitzar quines parts d'una imatge són més rellevants pel model durant el procés de reconeixement.
<br /><br /><br />

**FEEDBACK** <br />
•⁠  ⁠Utilitzar l'error de reconeixement de caràcters (Character Recognition Error) i Word Error Rate (WER), que representen l'accuracy, com a mètriques.<br />
•⁠  ⁠No utilitzar els Precision, Recall i F1-score, perquè son per la quantitat de classes que hi ha son mètriques massa denses .<br />
•⁠  ⁠Utilitzar HeatMap en comptes de les mètriques anteriors.<br />
•⁠  ⁠Resultat final: Model CRNN i CNN.
<br /> <br /> <br />




## **FASE 3**

**Pregunta 1:** Per a què es fa servir ctc en el codi CRNN?<br />
Aquest codi defineix una funció de pèrdua (loss function) anomenada ctc_lambda_func, que s'utilitza per entrenar el model perquè pugui reconèixer els noms escrits a mà. La funció de pèrdua CTC és crucial per a tasques de reconeixement de seqüències on no hi ha una alineació directa entre les dades d'entrada (per exemple, una seqüència d'imatges o un senyal d'àudio) i les etiquetes de sortida (per exemple, una seqüència de caràcters o paraules). Aquesta funció ajuda a entrenar el model perquè pugui aprendre a predir seqüències de sortida correctes, fins i tot quan la longitud de les seqüències d'entrada i sortida no coincideix.
⁠Durant la fase d'entrenament, la funció de pèrdua CTC s'utilitza per calcular la diferència entre les prediccions del model (y_pred) i les etiquetes reals (labels). Aquesta diferència es minimitza per ajustar els pesos del model i millorar les prediccions.

La CTC loss segmenta les imatges en infinitessimes parts per aconseguir una millor comprensió de les paraules escrites a mà. Aquesta funció calcula la probabilitat que els fragments successius formin part de la lletra o conjunt de lletres que s'està llegint, ajudant a identificar quines són les parts més probables de la paraula que s'està reconeixent. Això ajuda el model a generar prediccions més precises i acurades sobre el text que es vol llegir.
<br /><br />

**Pregunta 2:** Podria ser beneficios normalitzar la matriu de confusió?<br />
Sí, normalitzar la matriu de confusió pot ser molt útil perquè permet visualitzar les proporcions de prediccions correctes i incorrectes de manera més clara. 
Aquesta visualització ens permetrà veure millor quins caràcters són predits correctament i quins tenen més errors, ajudant a comprendre millor el rendiment del nostre model.
<br /><br />

**FEEDBACK** <br />
•⁠ Realitzar la conversió del model CNN a pytorch.
•⁠ Crear un model CNN ben definit i posteriorment aplicar RNN.
•⁠ Modificar el dataset de paraules, segmentant-les per lletres.
<br /> <br /> <br />


> [!NOTE]
> Com a convenient s'entèn que fa que el model aprengui millor.

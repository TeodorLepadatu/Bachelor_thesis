# Criptanaliza cifrului *Speck 32/64* folosind *Rețele Neuronale Convoluționale*

## Descrierea algoritmului de criptare *Speck 32/64*

### Parametri

Speck este un cifru bloc de tip ARX (Addition, Rotation, XOR). 
* **word size** = 16 biți
* **block** = $(L,R)$ = cuvântul criptat (care are 32 de biți) este împărțit în două subcuvinte de 16 biți
* **key** = număr pe 64 de biți
* **rotations** = numărul de biți rotiți la dreapta sau la stânga în funcția de criptare (valorile default sunt $\alpha = 7$ pentru rotația dreapta pe $L$ și $\beta = 2$ pentru rotația stânga pe $R$)
* **number of rounds** = numărul de subchei în care este împărțită cheia inițială

### Funcția de criptare

Definim:
* $ROR(L,\alpha)$ rotația la dreapta în $L$ cu $\alpha$ biți
* $ROL(R,\beta)$ rotația la stânga în $R$ cu $\beta$ biți
* $K_i$ subcheia rundei $i$, derivată din cheia inițială, având lungimea exactă de $w$ biți
* $\oplus$ operația $XOR$ pe biți

Acum, pentru **criptarea** mesajului folosim:

$$f(L,R) = (L',R')$$

unde:

$$L' = ((ROR(L,\alpha) + R) \bmod 2^{w}) \oplus K_i$$

$$R' = ROL(R,\beta) \oplus L'$$

Pentru **decriptare** folosim inversa funcției $f$:

$$f^{-1}(L',R') = (L,R)$$

unde:

$$R = ROR(R' \oplus L', \beta)$$

$$L = ROL(((L' \oplus K_i) - R) \bmod 2^{w}, \alpha)$$

## Atacul pentru decriptare

Atacatorul vede ciphertexte criptate în același mod, fără a cunoaște cheia folosită pentru criptarea lor. El vrea să găsească ultima subcheie cu care s-a criptat un mesaj. Odată ce găsește ultima cheie, repetă procesul până descoperă toată cheia secretă. Toate metricile de evaluare prezentate vor reflecta capacitatea algoritmilor de a returna această ultimă subcheie.

### Rețeaua neuronală convoluțională (CNN)

#### Definirea problemei
Construim o rețea neuronală convoluțională (CNN) care are rolul de *neural distinguisher*. Rețeaua va returna o probabilitate $p \in [0, 1]$ pentru a răspunde la întrebarea: "Provine perechea de texte cifrate $(C_1, C_2)$ din criptarea a două texte clare care respectă o anumită diferență fixă?”. În mod complementar, $1-p$ va reprezenta probabilitatea ca perechea respectivă să fie formată din secvențe de biți complet aleatoare.

#### Generarea datelor de antrenare
Pentru a antrena rețeaua folosim perechi de texte clare care diferă printr-o valoare fixă XOR, notată $\Delta P = (\Delta L, \Delta R)$. Procesul de generare a datelor pozitive (reale) presupune criptarea acestor perechi:

$$(L,R) \xrightarrow{encrypt} C_1$$

$$(L\oplus \Delta L, R\oplus \Delta R) \xrightarrow{encrypt} C_2$$

Astfel, un eșantion valid pentru rețea va fi format din combinarea perechilor de texte cifrate obținute: $(C_1, C_2)$. Setul de date complet este obținut echilibrând clasele: o jumătate este formată din exemple pozitive (perechi reale de criptare cu diferența dată), iar cealaltă jumătate din exemple negative (în care al doilea text cifrat $C_2$ este înlocuit cu o valoare generată uniform aleator).

#### Arhitectura rețelei
Arhitectura propusă este o rețea convoluțională reziduală formată dintr-un residual tower și un prediction head, arhitectură ce a fost simplificată și îmbunătățită pentru a reduce numărul de parametri fără pierderea acurateței.

**Structura Straturilor**
* **Input:** Un tensor tridimensional cu 3 canale, corespunzător caracteristicilor extrase din perechea $(C_1, C_2)$. Dimensiunea spațială este de 16 biți (conform arhitecturii Speck32/64).
* **Blocul rezidual:** Rețeaua integrează *depth* blocuri reziduale succesive. Fiecare bloc efectuează:
    * `Conv1d` (extindere de la 3 la 32 de canale, kernel = 3, padding = 1).
    * `BatchNorm1d` urmat de funcția de activare neliniară `ReLU`.
    * `Conv1d` (contractare de la 32 înapoi la 3 canale, kernel = 3, padding = 1).
    * `BatchNorm1d` urmat de `ReLU`.
    * O conexiune reziduală care adună intrarea originală a blocului la ieșirea acestuia, pentru a preveni vanishing gradient.
* **Prediction head**: Ieșirea turnului rezidual, de formă $(3, 16)$, este aplatizată într-un vector de 48 de elemente și trecută prin straturi fully connected:
    * `Linear` (48 $\rightarrow$ 64 de neuroni), `BatchNorm1d`, `ReLU`.
    * `Linear` (64 $\rightarrow$ 64 de neuroni), `BatchNorm1d`, `ReLU`.
    * `Linear` (64 $\rightarrow$ 1 neuron), urmat de funcția `Sigmoid` pentru a mapa ieșirea în probabilitatea binară dorită.

#### Parametrii și strategia de antrenare
Vom antrena modele pentru 5, 6 (având *depth* $= 10$) și 7 runde de criptare (având *depth* $= 1$), iar procesul de antrenare utilizează următoarea configurație hiperparametrică:
* **Funcția de pierdere (*Loss*):** Eroarea pătratică medie (`MSELoss`).
* **Optimizator:** `Adam`, cu o regularizare $L2$ și weight decay de $10^{-5}$.
* **Rata de învățare:** Planificator dinamic de tip `OneCycleLR`, cu o rată maximă de $10^{-3}$.
* **Setul de date:** $10^7$ eșantioane în total ($9 \cdot 10^6$ pentru antrenare, $10^6$ pentru validare).
* **Batch size**: 5000 de exemple per lot.
* **Durata:** Antrenarea rulează pentru 200 de epoci, reținându-se starea modelului cu cea mai bună acuratețe pe setul de validare.

Timpul total de antrenare a celor 3 rețele este de aproximativ 64 de ore folosind un *CPU i7 11th gen*, un *GPU GTX 1650* (cu *4GB VRAM*) și *16 GB RAM*. Același sistem a fost folosit și pentru metodele de atac ce vor fi prezentate ulterior.

#### Rezultatele antrenării
Modelul de **5 runde** are o acuratețe de 92.74%, cel de **6 runde** are 78.79 %, iar cel de **7 runde** 55.14 %.
     
Dacă am încerca să folosim aceeași strategie de antrenare pentru modele de 8 sau mai multe runde, acuratețea va fi în jurul valorii de 50 %, deci echivalent cu alegerea aleatorie a clasei, rezultat care nu poate fi folosit de niciunul dintre algoritmii de criptanaliză prezentați mai jos.

### Utilizarea probabilităților CNN

Odată antrenat, neural distinguisher-ul (DND-ul) nu este folosit izolat, ci ca o componentă centrală în faza de recuperare a subcheii (inferența rețelei pe date parțial decriptate). În continuare, vom prezenta două metode de agregare a probabilităților pentru a determina subcheia corectă.

#### Sum of Logits

Deoarece acuratețea rețelei pentru o singură pereche de texte cifrate este limitată, se utilizează structuri de texte cifrate generate pe baza unor biți neutri. Răspunsurile rețelei pentru toate perechile din structură sunt agregate pentru a formula un scor de încredere pentru fiecare cheie candidată.

**Notații:**
* $f_0(X) = P(real|X)$: probabilitatea returnată de CNN ca datele de intrare $X$ să provină dintr-o criptare reală.
* $X_i(K) = f^{-1}(C_i, K)$: rezultatul decriptării parțiale (cu o rundă) a perechii de texte cifrate $C_i$ folosind subcheia candidată $K$.
* $p_i(K) = f_0(X_i(K))$: probabilitatea estimată de rețea pentru perechea $i$ decriptată cu cheia $K$.
* $l_i(K) = \log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$: transformarea probabilității în *log-odds*.

**Descriere:**
Presupunem că dispunem de $n$ perechi de texte cifrate $(C_{i1}, C_{i2}), i=\overline{1,n}$, obținute dintr-o structură, și cunoaștem lungimea ultimei subchei. Pentru fiecare cheie candidată $K$ din spațiul de chei aferent, decriptăm parțial cele $n$ perechi. CNN-ul evaluează fiecare rezultat, oferind o probabilitate $p_i(K)$. Scorul total pentru subcheia candidată $K$ se calculează prin însumarea valorilor log-odds:

$$S(K) = \sum_{i=1}^{n}\log_2\left(\frac{p_i(K)}{1-p_i(K)}\right)$$

Valoarea maximă a scorului $S(K)$ va indica subcheia cea mai probabilă.

**Fundamentarea teoretică a metodei**

Această abordare este optimă sub două presupuneri stricte:

* CNN-ul este *Bayes-optimal*, adică probabilitatea prezisă reflectă perfect distribuțiile reale:

  $$P(real|X) = \frac{P_{real}(X)}{P_{real}(X) + P_{random}(X)}$$
  
  unde $P_{real}(X)$ este densitatea de probabilitate sub ipoteza că intrarea provine din distribuția cifrului, iar $P_{random}(X)$ este densitatea sub ipoteza unei distribuții uniforme.

* Cele $n$ exemple decriptate parțial $X_i(K)$ sunt independente condiționat de cheia $K$.

Pe baza primei presupuneri, deducem că:

$$\frac{f_0(X)}{1-f_0(X)} = \frac{P_{real}(X)}{P_{random}(X)} \iff \log_2\left(\frac{f_0(X)}{1-f_0(X)}\right) = \log_2\left(\frac{P_{real}(X)}{P_{random}(X)}\right)$$

Prin urmare, formula scorului devine echivalentă cu însumarea logaritmilor verosimilității (*log-likelihood*):

$$\forall i, K: \; l_i(K) = \log_2\left(\frac{p_i(K)}{1-p_i(K)}\right) = \log_2\left(\frac{P_{real}(X_i(K))}{P_{random}(X_i(K))}\right)$$

Dacă rețeaua este Bayes-optimă, maximizarea lui $S(K)$ este o decizie teoretic optimă pentru clasificarea secvențelor independente. În practică, deoarece CNN-ul este doar o aproximare a distribuției ideale, metoda oferă un doar estimator empiric, dar destul de robust având în vedere simplitatea acestui algoritm.

**Evaluarea algoritmului**

Pentru a ataca un sistem pe $n$ runde, vom folosi modelul antrenat pe $n-1$ runde. Vom considera că atacul a avut succes doar atunci când cheia adevărată se află printre un top de 32 de chei considerate de algoritm ca fiind cele mai probabile să fie cheia reală. Astfel, avem:

* Pentru atacul pe 6 runde am folosit modelul antrenat pe 5 runde și rata de succes este de 100 %, iar cheia reală se află în medie pe locul 1.50 în clasamentul cheilor date de algoritm.
* Pentru atacul pe 7 runde am folosit modelul antrenat pe 6 runde și rata de succes este de 100 %, iar cheia reală se află în medie pe locul 1.60 în clasamentul cheilor date de algoritm.
* Pentru atacul pe 8 runde am folosit modelul antrenat pe 7 runde și rata de succes este de 100 %, iar cheia reală se află în medie pe locul 5.10 în clasamentul cheilor date de algoritm.

#### Bayesian Key Search

Când decriptarea de probă se face pentru o singură rundă, ipoteza randomizării pentru chei greșite eșuează adesea, mai ales în cazul cifrurilor ușoare precum Speck32/64. Pentru a rezolva această problemă și a eficientiza căutarea, se utilizează algoritmul *Bayesian Key Search* (BKS), îmbunătățit prin preluarea și adaptarea acestuia pentru a garanta păstrarea cheilor optime.

**Precalcularea profilului (WKRP)**
Se generează un profil al răspunsului rețelei pentru chei greșite (WKRP - *Wrong Key Response Profile*). Pentru diverse diferențe dintre cheia reală și cheia de test ($\Delta k = k_i \oplus k$), se evaluează textele cifrate decriptate parțial. Transformând rezultatele DND în log-odds, obținem media $\mu_{\Delta k}$ și deviația standard $\sigma_{\Delta k}$ pentru fiecare diferență $\Delta k$ posibilă.

**Algoritmul de căutare iterativă**
Spre deosebire de forța brută (evaluarea întregului spațiu de chei), algoritmul BKS îmbunătățit execută $\ell$ iterații pentru a rafina succesiv un set restrâns de candidați. Procesul constă în următorii pași:

1. Se pornește cu un set $S$ format din $n_{cand}$ chei candidate. Pentru a preveni pierderea cheii reale din cauza fluctuațiilor statistice, dacă există o cheie optimă globală determinată în pașii sau loturile anterioare ($K_{best}$), aceasta este reținută forțat în setul curent de candidați.
2. Pentru fiecare cheie candidată $k_i \in S$ și fiecare pereche de texte cifrate $j$ din structură, decriptăm o rundă, trecem rezultatul prin DND pentru a obține probabilitatea $v_{j,k_i}$, iar apoi calculăm log-odds:
   
   $$z_{j,k_i} = \log_2\left(\frac{v_{j,k_i}}{1-v_{j,k_i}}\right)$$

3. Se calculează scorul mediu log-odds pentru fiecare cheie candidată $k_i$:
   
   $$m_{k_i} = \frac{1}{n_{cts}}\sum_{j=0}^{n_{cts}-1}z_{j,k_i}$$

4. Parcurgând întreg spațiul de chei posibile $k \in \mathcal{K}$, se calculează un scor de penalizare $\lambda(k)$, reprezentând distanța euclidiană ponderată:
   
   $$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

5. Se actualizează setul $S$ reținând cele $n_{cand}$ chei $k$ care minimizează scorul $\lambda(k)$ și se trece la următoarea iterație.

**Fundamentarea teoretică a metodei**
Eficiența metodei BKS se bazează pe următoarele presupuneri:
* Mediile log-odds empirice ($m_{k_i}$) urmează o distribuție normală dictată de diferența de cheie.
* Parametrii $\mu$ și $\sigma$ precalculați în tabelul WKRP reflectă fidel distribuția reală.

Considerând că media log-odds-urilor obținute cu cheia $k_i$ este distribuită normal în raport cu profilul cheii reale $k$:

$$m_{k_i} \sim \mathcal{N}(\mu_{k_i\oplus k}, \sigma_{k_i\oplus k}^2)$$

Funcția de verosimilitate (*likelihood*) pentru vectorul de medii observate $m$, condiționată de cheia corectă $k$, devine:

$$P(m|k) = \prod_{i=0}^{n_{cand}-1}\frac{1}{\sqrt{2\pi\sigma_{k_i\oplus k}^2}}e^{-\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{2\sigma_{k_i\oplus k}^2}}$$

Aplicând teorema lui Bayes (cu o distribuție *a priori* uniformă peste spațiul cheilor $P(k)$) obținem $P(k|m) \approx P(m|k)$. Trecând în domeniul logaritmic pentru a evita instabilitatea numerică și ignorând termenii constanți, maximizarea probabilității $\log_2 P(k|m)$ devine echivalentă cu minimizarea metricii noastre de eroare $\lambda(k)$:

$$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

**Evaluarea algoritmului**

Am folosit aceeași metodă de evaluare ca pentru *Sum of logits* și am obținut următoarele rezultate:

* Pentru atacurile pe 6 și 7 runde am obținut o rată de succes de 100 %, iar cheia adevărată este mereu prima în topul cheilor returnate de algoritm.
* Pentru atacul pe 8 runde, rata de succes este de 20 %, iar cheia reală este, în medie, pe locul 2.50.

### Ajustarea fină a modelului (*Fine-Tuning* cu exemple dificile)

Deși rețeaua neuronală descrisă anterior obține o acuratețe ridicată pe setul de date standard, performanța acesteia poate scădea în faza de atac (la recuperarea cheii). În generarea datelor standard, exemplele negative sunt create prin înlocuirea unui text cifrat valid cu date complet aleatoare. Totuși, în practică, în timpul atacului, rețeaua evaluează texte cifrate care au fost decriptate cu o subcheie candidată greșită. Aceste decriptări eronate nu produc un zgomot perfect aleator, ci păstrează anumite corelații structurale specifice cifrului, fenomen care slăbește distincția modelului. Pentru a rezolva această problemă, vom face un fine-tuning al modelului folosind exemple negative dificile. 

**Generarea exemplelor negative dificile**

Când generăm aceste exemple, vom simula exact scenariul întâlnit la căutarea cheii, urmărind următorii pași:

* Se generează și se criptează perechi de texte pentru $R$ runde folosind cheile corecte.
* Pentru un eșantion negativ, perechea obținută este criptată încă o rundă folosind subcheia corectă. Imediat după, perechea este decriptată o rundă, dar de această dată folosind o subcheie aleatoare.

Astfel, obținem un set de date care imită mai bine scenariile pe care modelul le va întâlni la inferență.

#### Rezultatele fine-tuning-ului

Am antrenat modelele pentru încă 20 de epoci folosind astfel de exemple negative, astfel crescând acuratețea lor pentru perechi de acest tip cu aproximativ 1%.

### Modele propuse și algoritmi hibrizi
Pentru a maximiza atât viteza de execuție, cât și rata de succes a atacului de decriptare, am implementat și evaluat o serie de strategii derivate. Acestea pleacă de la utilizarea directă a modelului fine-tuned și ajung până la algoritmi hibrizi. Algoritmii hibrizi combină avantajele de pre-filtrare ale metodei *Sum of Logits* (SoL) cu precizia rafinată a *Bayesian Key Search* (BKS), utilizând simultan cele două tipuri de rețele neuronale: modelul de bază (antrenat standard) și modelul cu ajustare fină (*fine-tuned*, notat FT).

Aceste strategii pot fi clasificate astfel:

* **Fine-Tuned BKS (FT BKS):**
  Această abordare reprezintă o îmbunătățire directă a algoritmului de bază. Execută algoritmul Bayesian Key Search pe întreg spațiul de chei ($2^{16} = 65536$ posibilități), dar folosește exclusiv modelul FT și profilul WKRP corespunzător acestuia, înlocuind modelul de bază.

* **SoL $\rightarrow$ Original BKS:**
  Evaluarea întregului spațiu de chei folosind direct BKS este o operațiune costisitoare computațional. Prin această metodă hibridă, utilizăm inițial tehnica SoL pe modelul de bază pentru a aproxima foarte rapid și a extrage doar un top restrâns de candidați (64 de chei). Setul obținut devine spațiul de căutare exclusiv pentru algoritmul BKS original, reducând masiv timpul de execuție, limitând în același timp numărul de candidați eronați care ar fi putut cauza rezultate fals-pozitive.

* **SoL $\rightarrow$ Fine-Tuned BKS:**
  Această tehnică respectă același principiu de restrângere a spațiului de căutare. După identificarea celor mai probabile 64 de chei folosind SoL și modelul de bază, rafinarea finală se face aplicând BKS exclusiv cu modelul FT. Metoda combină capacitatea de generalizare a modelului de bază cu acuratețea superioară a modelului FT în fața exemplelor negative dificile.

* **Ensemble BKS:**
  În cadrul acestei implementări de tip BKS, în locul inferenței pe un single model, neural distinguisher-ul acționează ca un ansamblu format din ambele modele (cel standard și cel FT). Răspunsurile acestora sunt ponderate pe baza inversului pătratelor funcției de pierdere obținute în procesul de validare:
  
  $$w_i = \frac{1}{\text{loss}_i^2}, \quad i \in \{1, 2\}$$
  
  Ponderile normalizate devin astfel:
  
  $$\alpha_1 = \frac{w_1}{w_1 + w_2}, \quad \alpha_2 = \frac{w_2}{w_1 + w_2}$$
  
  Probabilitatea agregată finală ($p_{final}$), care va fi ulterior transformată în log-odds pentru algoritmul bayesian, rezultă din suma ponderată a probabilităților individuale prezise de cele două modele:
  
  $$p_{final} = \alpha_1 \cdot p_{base} + \alpha_2 \cdot p_{FT}$$

* **SoL $\rightarrow$ Ensemble BKS:**
  Cea mai complexă arhitectură hibridă testează toate cele 65536 de chei folosind SoL și modelul de bază, iar cele 64 de chei care supraviețuiesc acestei etape de pre-filtrare sunt transmise mai departe către modelul de căutare *Ensemble BKS* pentru determinarea optimului global.

### Optimizarea algoritmului BKS

Deși algoritmul BKS aduce îmbunătățiri majore în rata de succes a recuperării cheii, calculul inițial al profilului WKRP (*Wrong Key Response Profile*) este o operațiune extrem de costisitoare din punct de vedere computațional. Pentru a construi acest profil de distribuții, rețeaua neuronală trebuie să evalueze milioane de perechi de texte cifrate pentru absolut toate cele $2^{16} = 65536$ de diferențe posibile de cheie ($\Delta k$). Pentru că acest profil rămâne identic pentru orice atac făcut de același model pe același număr de runde, am implementat un mecanism de caching offline. În loc să recalculăm profilul pentru fiecare nouă instanță sau scenariu de atac, tabelele WKRP (vectorii de medii $\mu$ și deviații standard $\sigma$) sunt generate o singură dată pentru fiecare model antrenat (atât cel de bază, cât și cel fine-tuned) și sunt serializate pe disc sub formă de fișiere. În faza de atac propriu-zisă doar se încarcă profilul în VRAM, operație mult mai rapidă decât calcularea profilul, dar și decât algoritmul *sum of logits* pentru care o astfel de precalculare este imposibilă.

Astfel, algoritmul BKS rulează inferența rețelei neuronale exclusiv pentru un set activ foarte mic de chei candidate ( $n_{cand} = 64$). Pentru a explora și a evalua restul spațiului de $65536$ de chei, BKS **nu** mai apelează rețeaua neuronală. În schimb, folosește valorile $\mu$ și $\sigma$ din cache-ul WKRP pentru a calcula distanța euclidiană ponderată $\lambda(k)$:

$$\lambda(k) = \sum_{i=0}^{n_{cand}-1}\frac{(m_{k_i}-\mu_{k_i\oplus k})^2}{\sigma_{k_i\oplus k}^2}$$

Această evaluare bayesiană se reduce la operații matematice vectorizate elementare pe tensori, care sunt executate aproape instant pe arhitectura masiv paralelă a unui GPU, ocolind complet necesitatea unor noi inferențe cu CNN-ul.

## Evaluarea și rezultatele metodelor de criptanaliză

Pentru a compara eficiența metodelor propuse, evaluarea a fost realizată printr-un cadru de testare automatizat care simulează scenarii de atac pe 6, 7 și 8 runde. 

### Metodologia de evaluare
Procesul de evaluare funcționează printr-o buclă continuă, în care fiecare iterație reprezintă o nouă provocare. Pentru fiecare provocare sunt generați următorii parametri:
* O subcheie țintă generată aleatoriu, reprezentând obiectivul atacului ce trebuie recuperat.
* Un număr specific de structuri de texte clare și cifrate: 32 de structuri pentru atacul pe 6 runde, 64 de structuri pentru 7 runde și 128 de structuri pentru 8 runde.

În cadrul fiecărei iterații, textele cifrate sunt transmise secvențial către șase metode distincte de recuperare a cheii:
1. **M1 (Original BKS):** Metoda de bază, folosind algoritmul BKS cu modelul antrenat standard.
2. **M2 (Fine-Tuned BKS):** Algoritmul BKS evaluat exclusiv cu modelul fine-tuned.
3. **M3 (SoL $\rightarrow$ Original BKS):** Reducerea spațiului de căutare la 64 de candidați folosind metoda *Sum of Logits* (SoL) și evaluarea acestora cu BKS-ul original.
4. **M4 (SoL $\rightarrow$ Fine-Tuned BKS):** Reducerea spațiului prin SoL, urmată de evaluarea BKS folosind modelul ajustat fin.
5. **M5 (Ensemble BKS):** Algoritmul BKS utilizând un ansamblu format din ambele modele (standard și ajustat).
6. **M6 (SoL $\rightarrow$ Ensemble BKS):** Spațiu redus prin SoL la primele 64 de chei, evaluat ulterior cu metoda Ensemble BKS.

### Metrici înregistrate și criteriul de oprire
Pentru fiecare metodă și fiecare provocare, algoritmul măsoară timpul de execuție și verifică dacă subcheia prezisă se potrivește perfect cu subcheia țintă reală. Pe baza acestor date, se calculează și se actualizează dinamic acuratețea globală (numărul de predicții corecte raportat la numărul total de rulări) și timpul mediu de execuție. Spre deosebire de evaluările anterioare, vom considera că atacul a avut succes doar dacă cheia corectă este prima în topul cheilor prezise de algoritm. Astfel, acuratețea va fi considerabil mai mică decât în analiza anterioară.

Pentru a asigura o evaluare relevantă din punct de vedere statistic, mediul de testare impune un număr de minimum 10 rulări înainte de a verifica vreo condiție de dominanță. Bucla infinită de evaluare se oprește doar în momentul în care cel puțin una dintre metodele propuse (M2 -- M6) demonstrează o dominanță clară asupra metodei de bază (M1). Această dominanță este definită prin îndeplinirea uneia dintre următoarele două condiții:
* **Acuratețe superioară:** Acuratețea metodei propuse este strict mai mare decât acuratețea metodei de bază (M1).
* **Dominanță la viteză:** Metoda propusă obține o acuratețe mai mare sau egală cu cea a metodei M1 (ambele având o acuratețe strict mai mare ca 0), dar înregistrează un timp mediu de execuție strict mai mic.

În momentul în care oricare dintre aceste criterii de succes este atins, rularea scenariului se oprește, mediul afișează rezumatul final cu timpii și acuratețile tuturor celor 6 metode, iar întregul istoric al seturilor de date aferente (până la o limită fixă de 1000 de provocări) este salvat automat pe disc.

### Rezultatele atacurilor

Pentru a valida performanța metodelor propuse, bucla de testare a fost rulată pentru cele trei scenarii distincte de atac (6, 7 și 8 runde). În toate cele trei cazuri, algoritmii hibrizi și cei bazați pe modelul fine-tuned au demonstrat o superioritate clară față de metoda originală de referință (M1 - Orig BKS), rezultatele menținând un tipar consistent de dominanță. Această metodă de evaluare fiind constructivă, am găsit și un set de date în care cel puțin unul dintre algoritmii propuși este mai performant decât algoritmul de referință.

#### Analiza acurateței
În primele două scenarii de atac (6 și 7 runde), metodele M2 (FT BKS), M4 (SoL $\rightarrow$ FT BKS) și M5 (Ensemble BKS) au atins rapid o acuratețe de 100\%, depășind performanța de 90\% a algoritmului de bază (M1). Mai mult, în cel de-al treilea scenariu (8 runde), considerat un caz extrem din cauza degradării probabilităților și a zgomotului masiv din date, metodele pur bayesiene (M1, M2, M5) au înregistrat o rată de succes de 0\%. În contrast direct, metodele care folosesc *Sum of Logits* pentru pre-filtrare (M3, M4, M6) au reușit să recupereze cheia corectă în 20\% din cazuri, demonstrând o robustețe net superioară a arhitecturilor hibride.

#### Analiza timpului de execuție
Diferențele masive ale timpului de execuție confirmă avantajul teoretic al mecanismului de caching offline explicat anterior. Metodele BKS directe (M1 și M2) sunt extrem de rapide, finalizând o provocare în sub o secundă (în medie 0.54s -- 0.78s). Metoda M5 (*Ensemble BKS*) necesită aproximativ dublul acestui timp (1.06s -- 1.52s), o creștere logică și eficientă, având în vedere că prelucrează simultan răspunsurile a două rețele neuronale diferite. 

Pe de altă parte, metodele care integrează componenta SoL (M3, M4, M6) sunt constrânse de necesitatea de a efectua inferența CNN pe întregul spațiu de 65536 de chei. Acest aspect duce la timpi de execuție semnificativ mai mari, cuprinși între 155 și 342 de secunde per provocare, ilustrând perfect compromisul viteză-complexitate, dar cu o performanță mai ridicată în unele cazuri.
În al doilea rând, optimizarea prin caching offline a profilului WKRP elimină necesitatea inferenței repetate, transformând algoritmul *Bayesian Key Search* (BKS) într-un proces mult mai rapid decât abordarea clasică *Sum of Logits* (SoL).

Principala contribuție o reprezintă algoritmii hibrizi propuși (M2 -- M6), care oferă un compromis ideal între viteză și acuratețe. Pentru atacuri pe 6 și 7 runde, metodele *FT BKS* și *Ensemble BKS* obțin o acuratețe de 100\% într-un timp de maxim 2 secunde. Pentru scenariul de 8 runde, unde metodele clasice eșuează complet (0\% succes) pe setul de date găsit, utilizarea pre-filtrării SoL combinată cu analiza *Ensemble* a reprezentat singura soluție viabilă, recuperând cheia în 20\% din cazuri.

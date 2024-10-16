# Lyhyt kuvaus

Tässä koodissa toteutettu Gaussin Naive Bayes -classifier on hyvä esimerkki yksinkertaisesta ja tehokkaasta koneoppimismenetelmästä. Naive Bayes perustuu todennäköisyyslaskentaan ja erityisesti Bayesin teoreemaan, ja se olettaa, että kaikki ominaisuudet ovat toisistaan riippumattomia. Tämä oletus on usein naiivi, mutta menetelmä toimii siitä huolimatta hyvin monissa tilanteissa, erityisesti silloin, kun datasetti on pieni ja ominaisuuksia on paljon suhteessa datan määrään.

Naive Bayesin merkittävimpiä etuja verrattuna muihin koneoppimismenetelmiin, kuten tukivektorikoneisiin (SVM) tai satunnaismetsiin (Random Forest), ovat sen yksinkertaisuus ja nopeus. Koska Gaussin Naive Bayes perustuu yksinkertaisiin tilastollisiin laskelmiin, se on resurssitehokas ja soveltuu hyvin suuriin datamääriin. Tämä malli toimii hyvin myös silloin, kun datan ominaisuudet noudattavat likimain normaalijakaumaa, kuten tässä esimerkissä oletetaan. Yksinkertaisuutensa ansiosta se on erittäin helppokäyttöinen ja helposti tulkittava.

Toisaalta Naive Bayesilla on myös rajoituksia verrattuna monimutkaisempiin menetelmiin, kuten satunnaismetsiin tai neuroverkkoihin. Satunnaismetsät pystyvät oppimaan monimutkaisempia riippuvuuksia datasta, ja neuroverkot voivat mallintaa erittäin monimutkaisia ilmiöitä suurista datamääristä. Naive Bayes ei pysty mukautumaan monimutkaisiin, ei-lineaarisiin suhteisiin ominaisuuksien välillä, ja riippumattomuusoletus voi aiheuttaa merkittäviä epätarkkuuksia, jos ominaisuudet ovat vahvasti toisistaan riippuvaisia.

Gaussin Naive Bayes on käytännössä erinomainen työkalu tilanteissa, joissa tarvitaan yksinkertainen ja nopeasti koulutettava malli, joka antaa hyviä tuloksia kohtuullisella vaivalla. Erityisesti tekstiluokittelussa, jossa ominaisuudet (kuten sanat) ovat usein suhteellisen riippumattomia toisistaan, Naive Bayes toimii usein odottamattoman hyvin. Tämä harjoitus auttoi minua ymmärtämään paremmin, miten erilaiset koneoppimismallit eroavat toisistaan ja milloin yksinkertainen menetelmä voi olla tehokkain vaihtoehto

# videot

[Naive Bayes Classifier Python Tutorial](https://www.youtube.com/watch?v=S0LYNF1Ftuk&t=451s)
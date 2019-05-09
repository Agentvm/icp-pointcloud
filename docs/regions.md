## Kriterien für Regionen

-   xyz gut
-   xy gut
-   xz gut
-   yz gut
-   y gut
-   Wald
-   Acker
-   Wasser
-   Fahrbahn
-   Gebäude abgerissen/ neu gebaut


### Notizen

-   100m x 100m + Rand für Normalenvektorbestimmung
-   Ausschneiden, CC-Werte abspeichern (siehe unten)
-   Alle Details zur erzeugten Wolke notieren
-   Matching: 20 + Punktpaare pro Kachel


## Cloud Compare Prozess zum Extrahieren eines Punktwolkensegmentes

-   Verschiebung beim Einladen der Wolke minimieren
-   Wähle "Set Top View"
-   Wähle "Tools>Segmentation>Cross Section"
-   Passe die Parameter per visueller Verschiebung oder im Fenster an
-   Wähle "Extract contour as polyline" im Fenster, wähle Flat dimension = 'z', Max edge length = 100, bestätige den Dialog
-   Beende Cross Section (Fenster schließen)
-   Speichere zur Wiederholbarkeit die erzeugte Polyline unter dem Namen der Punktwolke ab (Endung .bin)
-   Wähle "Segment" in der oberen Leiste, oder "Tools>Segmentation>Extract Segment"
-   Wähle "Import polyline from DB for segmentation" im neuen Fenster und wähle die erzeugte Polyline
-   Wähle "Confirm Segmentation" im Fenster
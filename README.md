# Dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Preprocessing danych + balans klas
Umieść w głównym katalogu projektu katalog dataset z katalogami train, test, val. (Każdy z nich będzie posiadał wewnątrz NORMAL, PMEUMONIA)

Następnie w zależności od wybranego problemu 2 klasy lub 3 klasy - uruchom skrypt z katalogu `dataset_procesing`.

# Trening sieci
W katalogu `src` są trzy pliki po jednym dla kazego typu sieci. Na górze każego pliku zdefiniowany jest path do wybranego zbioru danych, należy jedynie go przełączyć - reszta powinna zadziałać.


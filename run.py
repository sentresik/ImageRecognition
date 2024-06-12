# zaladowanie bibliotek - wykorzystanie technologii SVM i HOG

# zaladowanie danych


# normalizacja danych


# wrzucamy dane do HOG �eby u�atwi� SVM prace


# podzia� danych na treningowe i testowe


# trening modelu - model SVM i trenowanie na zbiorze treningowy


# ocena modelu - dane testowe i sprawdzenie wynikow.


# wizualizacja wynikow - wykresy i dane


# petla iteracyjna - powtarzanie i udoskonalanie modelu

import subprocess

def main():
    try:
        # Uruchomienie skryptu process_datasets.py
        subprocess.run(['python', 'process_datasets.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running process_datasets.py: {e}")

if __name__ == "__main__":
    main()
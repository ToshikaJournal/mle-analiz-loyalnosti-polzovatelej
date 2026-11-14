from src.processing import DataProcessor

def main():
    print("Запуск анализа лояльности пользователей Яндекс Афиши...")
    print()
    
    # Инициализируем класс
    processor = DataProcessor()
    
    # Запускаем полный пайплайн
    processor.run_pipeline()
    
    print(" Проект завершён!")

if __name__ == '__main__':
    main()
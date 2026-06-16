import os
import glob

# Вкажіть шляхи до ваших папок із розміткою та зображеннями
# (Приклад для папки train, аналогічно треба буде зробити для val та test)
labels_dir = 'dataset/valid/labels'
images_dir = 'dataset/valid/images'

# ID класів з оригінального PlantDoc, які ми залишаємо як релевантні (Яблуня, Вишня, Персик, Виноград)
classes_to_keep = {'0', '1', '2', '6', '10', '28', '29'}

# Підтримувані формати зображень
image_extensions = ['.jpg', '.jpeg', '.png']

def process_dataset():
    # Отримуємо список усіх текстових файлів з розміткою
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    deleted_files_count = 0
    updated_files_count = 0
    
    for txt_path in txt_files:
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            
        new_lines = [] # Виправлено: тепер це порожній список
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            class_id = parts[0] # Виправлено: беремо перший елемент рядка (ID класу)
            
            # Якщо об'єкт належить до цільових дерев
            if class_id in classes_to_keep:
                # Змінюємо ID класу на '0' (наш єдиний макро-клас 'leaf')
                parts[0] = '0' # Виправлено: переприсвоюємо саме перший елемент
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)
                
        # Якщо після фільтрації не залишилося жодного потрібного листка
        if len(new_lines) == 0:
            os.remove(txt_path) # Видаляємо порожній файл розмітки
            
            # Шукаємо та видаляємо відповідну фотографію, щоб уникнути помилок навчання
            base_name = os.path.splitext(os.path.basename(txt_path))[0] # Виправлено: беремо лише ім'я без розширення
            for ext in image_extensions:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    break
                    
            deleted_files_count += 1
        else:
            # Якщо рамки є, перезаписуємо файл з новими класами
            with open(txt_path, 'w') as file:
                file.writelines(new_lines)
            updated_files_count += 1
            
    print("Обробку завершено!")
    print(f"Оновлено файлів з цільовими листками (клас 0): {updated_files_count}")
    print(f"Видалено файлів (зайві рослини або фон): {deleted_files_count}")

if __name__ == '__main__':
    process_dataset()
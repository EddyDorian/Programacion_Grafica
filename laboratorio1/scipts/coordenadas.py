import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
image_path = "elefante2-picaai.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Invertir la imagen para que las figuras negras sean blancas (facilita la detección)
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Obtener los vértices de los contornos
vertices = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)  # Ajustar precisión de la aproximación
    approx = cv2.approxPolyDP(contour, epsilon, True)
    for point in approx:
        vertices.append(point[0])  # Extraer coordenadas (x, y)

# Convertir coordenadas a rango [-1,1]
height, width = image.shape
vertices = np.array(vertices, dtype=np.float32)
vertices[:, 0] = 2 * (vertices[:, 0] / width) - 1  # Escalar x
vertices[:, 1] = 2 * (vertices[:, 1] / height) - 1  # Escalar y

# Invertir eje Y para corregir orientación
vertices[:, 1] = -vertices[:, 1]

# Mostrar los vértices sobre la imagen
plt.imshow(thresh, cmap="gray")
plt.scatter((vertices[:, 0] + 1) * width / 2, (1 - vertices[:, 1]) * height / 2, color='red', s=10)
plt.show()

# Guardar los vértices en un archivo
with open("vertices.txt", "w") as f:
    for vertex in vertices:
        f.write(f"{vertex[0]}, {vertex[1]},\n")

# Imprimir los vértices en consola
print("Coordenadas de los vértices:")
for vertex in vertices:
    print(f"({vertex[0]}, {vertex[1]})")

#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 800;
const int POINT_COUNT = 1000;
const float PI = 3.1415926535f;

void display() {
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f); // Light gray background
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0.0f, 0.0f, 0.0f); // Black symbol
    glPointSize(2.0f);

    glBegin(GL_POINTS);
    for (int i = 0; i < POINT_COUNT; ++i) {
        float t = 2.0f * PI * (static_cast<float>(i) / POINT_COUNT);
        
        // Gerono's Lemniscate formula
        // Map x, y to a -1 to 1 range suitable for standard OpenGL coordinates
        float x = sin(t);
        float y = sin(t) * cos(t);

        glVertex2f(x * 0.8f, y * 0.8f); // Slightly scale down
    }
    glEnd();

    glFlush();
}

int main(void) {
    GLFWwindow* window;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGL Infinity Symbol", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create window" << std::endl;
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

#pragma once

#include <SDL2/SDL.h>
#include <vector>
#include <iostream>

#include "camera.hpp"

struct FPSCounter {
    Uint32 frameCount = 0;
    Uint32 lastTime = 0;
    float currentFPS = 0.0f;
    
    void update() {
        frameCount++;
        Uint32 currentTime = SDL_GetTicks();
        
        // Update FPS every second
        if (currentTime - lastTime > 1000) {
            currentFPS = frameCount * 1000.0f / (currentTime - lastTime);
            printf("\rFPS: %.1f", currentFPS);
            fflush(stdout);
            
            frameCount = 0;
            lastTime = currentTime;
        }
    }
};

struct BoundingSphere {
    float3 center;
    float radius;
};

struct MouseState {
    bool leftButtonDown = false;
    int lastX = 0;
    int lastY = 0;
    float totalRotationX = 0.0f;  // Track total rotation
    float totalRotationY = 0.0f;
    float lastRotationX = 0.0f;   // Track last frame's rotation
    float lastRotationY = 0.0f;
};

BoundingSphere calculateBoundingSphere(const std::vector<float3>& vertices) {
    BoundingSphere sphere;
    
    // Calculate center as average of all vertices
    sphere.center = make_float3(0.0f, 0.0f, 0.0f);
    for (const auto& v : vertices) {
        sphere.center.x += v.x;
        sphere.center.y += v.y;
        sphere.center.z += v.z;
    }
    sphere.center.x /= vertices.size();
    sphere.center.y /= vertices.size();
    sphere.center.z /= vertices.size();
    
    // Calculate radius as maximum distance from center to any vertex
    sphere.radius = 0.0f;
    for (const auto& v : vertices) {
        float dx = v.x - sphere.center.x;
        float dy = v.y - sphere.center.y;
        float dz = v.z - sphere.center.z;
        float dist = sqrt(dx*dx + dy*dy + dz*dz);
        sphere.radius = std::max(sphere.radius, dist);
    }
    
    return sphere;
}

void handleMouseEvent(SDL_Event& event, MouseState& mouseState, short WINDOW_WIDTH, short WINDOW_HEIGHT) {
    switch(event.type) {
        case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_LEFT) {
                mouseState.leftButtonDown = true;
                mouseState.lastX = event.button.x;
                mouseState.lastY = event.button.y;
                // Store current rotation when starting drag
                mouseState.lastRotationX = mouseState.totalRotationX;
                mouseState.lastRotationY = mouseState.totalRotationY;
            }
            break;
            
        case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_LEFT) {
                mouseState.leftButtonDown = false;
            }
            break;
            
        case SDL_MOUSEMOTION:
            if (mouseState.leftButtonDown) {
                // Convert mouse position to rotation angles
                mouseState.totalRotationX = mouseState.lastRotationX + 
                    (event.motion.y - mouseState.lastY) * (2.0f * M_PI / WINDOW_HEIGHT);
                mouseState.totalRotationY = mouseState.lastRotationY + 
                    (event.motion.x - mouseState.lastX) * (2.0f * M_PI / WINDOW_WIDTH);
            }
            break;
    }
}

// =============================================================
// SDLApp class
// =============================================================
class SDLApp
{
public:
    SDLApp(short windowWidth, short windowHeight)
        : m_windowWidth(windowWidth), 
          m_windowHeight(windowHeight) 
    {
        // The constructor doesn’t necessarily do the init; 
        // you can do it in init() so you can handle errors.
    }

    ~SDLApp() {
        // Clean up in reverse order of creation
        if (m_texture)   SDL_DestroyTexture(m_texture);
        if (m_renderer)  SDL_DestroyRenderer(m_renderer);
        if (m_window)    SDL_DestroyWindow(m_window);
        SDL_Quit();
    }

    bool init(const char* title = "Gaussian Splats Viewer") {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create the SDL window
        m_window = SDL_CreateWindow(
            title,
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            m_windowWidth, m_windowHeight,
            SDL_WINDOW_SHOWN
        );
        if (!m_window) {
            std::cerr << "Window creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create the renderer
        m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED);
        if (!m_renderer) {
            std::cerr << "Renderer creation failed: " << SDL_GetError() << std::endl;
            return false;
        }

        // Create the texture
        m_texture = SDL_CreateTexture(
            m_renderer,
            SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STREAMING,
            m_windowWidth, m_windowHeight
        );
        if (!m_texture) {
            std::cerr << "Texture creation failed: " << SDL_GetError() << std::endl;
            return false;
        }
        return true;
    }

    // Poll and process events.  
    // You can keep the handleMouseEvent logic inside or outside the class.
    bool processEvents(MouseState& mouseState, bool& running, SDL_Event& event)
    {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            handleMouseEvent(event, mouseState, m_windowWidth, m_windowHeight);
        }
        return true; // keep running
    }

    // Call this once per frame to update the texture and present
    void renderFrame(std::vector<Uint32>& pixels, 
                     FPSCounter& fpsCounter, 
                     const OrthoCameraParams& camera)
    {
        // Update the texture with the latest pixel buffer
        SDL_UpdateTexture(m_texture, NULL, pixels.data(), 
                          camera.imageWidth * sizeof(Uint32));

        // Clear, copy, and present
        SDL_RenderClear(m_renderer);
        SDL_RenderCopy(m_renderer, m_texture, NULL, NULL);
        SDL_RenderPresent(m_renderer);

        // Update FPS
        fpsCounter.update();
    }

private:
    short          m_windowWidth  = 0;
    short          m_windowHeight = 0;
    SDL_Window*    m_window       = nullptr;
    SDL_Renderer*  m_renderer     = nullptr;
    SDL_Texture*   m_texture      = nullptr;
};


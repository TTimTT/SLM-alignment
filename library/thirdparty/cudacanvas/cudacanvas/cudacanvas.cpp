#include <GLFW/glfw3.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_gl_interop.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <queue>

#define GL_TEXTURE_SWIZZLE_A 0x8E45
#define GL_TEXTURE_SWIZZLE_B 0x8E44
#define GL_TEXTURE_SWIZZLE_G 0x8E43
#define GL_TEXTURE_SWIZZLE_R 0x8E42
#define GL_TEXTURE_SWIZZLE_RGBA 0x8E46

namespace py = pybind11;


class CudaGLStreamer {

  public:
    CudaGLStreamer() : rendering(false), render_ready(false), refresh_period(1.0/60.0) {

        std::cout << "Using GLFW: " << glfwGetVersionString() << std::endl;
        setImage(torch::zeros({28, 28}, torch::kU8).to(torch::kCUDA));
    }

    ~CudaGLStreamer() {
        stopRender();
    }

    void cleanUp() {
        if(cudaResource)
            cudaGraphicsUnregisterResource(cudaResource);

        if(textureID)
        glDeleteTextures(1, &textureID);

        if(window)
            glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createWindows() {
        // Initialize GLFW in the constructor
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        // Remove decoration of window
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        window = glfwCreateWindow(source_image.size(1), source_image.size(0), "Multi-Window Example", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwMakeContextCurrent(window);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // 0x812F is typically the value for GL_CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, source_image.size(1), source_image.size(0), 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
        //glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);

        GLenum glError = glGetError();
        if (glError != GL_NO_ERROR) {
            fprintf(stderr, "OpenGL error before registering texture: %d\n", glError);
        }

        cudaError_t cudaStatus = cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGraphicsGLRegisterImage failed: %s\n", cudaGetErrorString(cudaStatus));
        }
        
        glEnable(GL_TEXTURE_2D);
        glfwMakeContextCurrent(NULL);
    }

    void setWindowFull(int monitor_id = 0) {

        // Set autofocusing when creating the window
        //glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);

        // Get all possible monitors
        int count;
        GLFWmonitor** monitors = glfwGetMonitors(&count);

        if(count == 0) {
            glfwTerminate();
            throw std::runtime_error("Failed to find a monitor");
        }

        if((monitor_id >= count) || (monitor_id < 0))
        {
            glfwTerminate();
            throw std::runtime_error("Invalid monitor ID! There are " + std::to_string(count) +" total monitors");   
        }

        // Create window on the desired monitor
        GLFWmonitor* monitor = monitors[monitor_id];
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

        int xpos, ypos;
        glfwGetMonitorPos(monitor, &xpos, &ypos);
        glfwSetWindowPos(window, xpos, ypos);
        //glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        refresh_period = std::min(refresh_period, 1.0/mode->refreshRate);
    }

    void restoreWindow(int monitor_id = 0) {
        // Get all possible monitors
        //glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
        int count, xpos, ypos;
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        
        GLFWmonitor* monitor = monitors[monitor_id];
        glfwGetMonitorPos(monitor, &xpos, &ypos);
        glfwSetWindowPos(window, xpos, ypos);
        glfwRestoreWindow(window);
        
        glfwMakeContextCurrent(window);
        glfwShowWindow(window);
        glfwSwapBuffers(window);

        glfwSwapInterval(1);
        glfwPollEvents();
        glfwMakeContextCurrent(NULL);
    }

    void render() {
        GLenum glError;
        while(rendering) {
            glfwMakeContextCurrent(window);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindTexture(GL_TEXTURE_2D, textureID);
            //glColorMask(GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);
            cudaError_t cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }

            cudaStatus = cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaResource, 0, 0);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGraphicsSubResourceGetMappedArray failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }
            if(!this->img_ptr.empty() && this->render_ready) {
                cudaStatus = cudaMemcpy2DToArray(texturePtr, 0, 0, reinterpret_cast<void*>(this->img_ptr.front()), source_image.size(1) * sizeof(uint8_t), source_image.size(1) * sizeof(uint8_t), source_image.size(0), cudaMemcpyDefault);
                
                if(this->img_ptr.size() > 1)
                    this->img_ptr.pop();
                
            } else {
                cudaStatus = cudaMemcpy2DToArray(texturePtr, 0, 0, reinterpret_cast<void*>(source_image.data_ptr()), source_image.size(1) * sizeof(uint8_t), source_image.size(1) * sizeof(uint8_t), source_image.size(0), cudaMemcpyDefault);
            }

            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy2DToArray failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }

            cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, 0);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(cudaStatus));
                return;
            }
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(vertex_coords[0][0],  vertex_coords[1][0]);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(vertex_coords[0][1],  vertex_coords[1][1]);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(vertex_coords[0][2],  vertex_coords[1][2]);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(vertex_coords[0][3],  vertex_coords[1][3]);
            glEnd();
            glBindTexture(GL_TEXTURE_2D, 0);
            glFinish();
            
            glfwSwapBuffers(window);
            glfwSwapInterval(1);
            glfwPollEvents();

            glError = glGetError();
            if (glError != GL_NO_ERROR) {
                std::cerr << "OpenGL error " << glError <<  " ";
            }
            
            if (glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE)) {
                glfwTerminate();
                exit(EXIT_SUCCESS);
            }
        }
    }


    void setImage(torch::Tensor image) {
        if (!image.is_cuda()) {
            throw std::runtime_error("Input tensor is not on CUDA. Please provide a CUDA tensor.");
        }
        source_image = image.to(torch::kU8);
        this->setImagePTR(reinterpret_cast<uint64_t>(source_image.data_ptr()));
    }

    void setImagePTR(uint64_t img_ptr) {

        this->img_ptr.push(img_ptr);
    }

    uint64_t getImgPTR() {
        return this->img_ptr.front();
    }

    void printQueue()
    {
        std::queue<uint64_t> copy = this->img_ptr;

        std::cout << "Content of DisplayGL FIFO:" << std::endl;
        while (!copy.empty())
        {
            std::cout << copy.front() << " ";
            copy.pop();
        }
        std::cout << std::endl;
    }

    void clearPTRQueue() {
        std::queue<uint64_t> empty;
        std::swap(this->img_ptr, empty);
    }

    double getRefreshPeriod(void) const {
        return refresh_period;
    }

    void setFlipXY(bool x_flip, bool y_flip) {
        if(x_flip) {
            for(auto& vertex : vertex_coords[0])
                vertex = -vertex;
        }
        if(y_flip) {
            for(auto& vertex : vertex_coords[1])
                vertex = -vertex;
        }
    }

    void setRenderReady(bool is_ready) {
        render_ready = is_ready;
    }

    void startRender() {
        rendering = true;
        setRenderReady(false);
        render_thread = std::thread(&CudaGLStreamer::render, this);
    }

    void pauseRender() {
        rendering = false;
        setRenderReady(false);
        render_thread.join();
    }

    void stopRender() {
        rendering = false;
        setRenderReady(false);
        render_thread.join();
        cleanUp();
    }

  private:
    const char* description;
    double refresh_period;
    std::thread render_thread;
    bool rendering;
    bool render_ready;

    GLFWwindow* window;
    cudaGraphicsResource_t cudaResource;
    GLuint textureID;

    cudaArray_t texturePtr;
    torch::Tensor source_image;
    std::queue<uint64_t> img_ptr;

    GLfloat vertex_coords[2][4] = {{-1.0f, 1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, -1.0f, -1.0f}};
};

PYBIND11_MODULE(cudaGLStream, m)
{
    py::class_<CudaGLStreamer>(m, "CudaGLStreamer")
        .def(py::init<>())
        .def("set_flip_xy", &CudaGLStreamer::setFlipXY)
        .def("set_image", &CudaGLStreamer::setImage)
        .def("set_imagePTR", &CudaGLStreamer::setImagePTR)
        .def("get_imgPTR", &CudaGLStreamer::getImgPTR)
        .def("clear_PTRQueue", &CudaGLStreamer::clearPTRQueue)
        .def("print_queue", &CudaGLStreamer::printQueue)
        .def("get_refresh_period", &CudaGLStreamer::getRefreshPeriod)
        .def("create_windows", &CudaGLStreamer::createWindows)
        .def("set_windowfull", &CudaGLStreamer::setWindowFull)
        .def("restore_window", &CudaGLStreamer::restoreWindow)
        .def("set_render_ready", &CudaGLStreamer::setRenderReady)//, py::call_guard<py::gil_scoped_release>())
        .def("start_render", &CudaGLStreamer::startRender)//, py::call_guard<py::gil_scoped_release>())
        .def("pause_render", &CudaGLStreamer::pauseRender)
        .def("stop_render", &CudaGLStreamer::stopRender)
        .def("clean_up", &CudaGLStreamer::cleanUp);
}

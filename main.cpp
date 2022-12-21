#include <fstream>  // config file reading
#include <iostream>
#include <regex>    // config file reading
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "CImg.h"


///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int BLOCK_SIZE = 32;

const std::string CONFIG_FILENAME = "input.conf";

constexpr char KEY_VALUE_DELIMITER = ':';
constexpr char TUPLE_DELIMITER = ',';
constexpr char COMMENT_INDICATOR = '#';

const std::string KEY_GRID_CELLS = "NumGridCells";
const std::string KEY_GRID_SIZE = "GridCellSize";
const std::string KEY_TIMESTEP = "Timestep";
const std::string KEY_RUNTIME = "TotalRuntime";
const std::string KEY_SMOKEGEN_LOC = "SmokeGeneratorLocation";
const std::string KEY_SMOKEGEN_VAL = "SmokeGeneratorValue";
const std::string KEY_ADVECTION = "AdvectionConstant";
const std::string KEY_EDDY = "EddyDiffusivities";


///////////////////////////////////////////////////////////////////////////////
// Structs
///////////////////////////////////////////////////////////////////////////////

struct InputConfig
{
    int gridCellsX;
    int gridCellsY;
    float gridCellSizeX;
    float gridCellSizeY;

    float timestep;
    float totalRuntime;

    int smokeGenLocationX;
    int smokeGenLocationY;
    float smokeGenValue;

    float advectionConstant;
    float eddyDiffusivityX;
    float eddyDiffusivityY;
};


///////////////////////////////////////////////////////////////////////////////
// Macros
///////////////////////////////////////////////////////////////////////////////

// handy error handler I yoinked from a CUDA textbook
void HandleError(const hipError_t err, const char *file, const int line)
{
    if (err == hipSuccess) return;

    std::cout << "Error encountered in file " << file << " at line " << line << ":\n";
    std::cout << "    " << hipGetErrorName(err) << ": " << hipGetErrorString(err) << "\n";
}

#define HANDLE_ERROR(err) HandleError(err, __FILE__, __LINE__);


///////////////////////////////////////////////////////////////////////////////
// Helper function declarations
///////////////////////////////////////////////////////////////////////////////
//
const InputConfig GetInputConfig(const std::string& inputFilename);
const std::vector<std::string> WhitespaceTokenizer(const std::string& line);


///////////////////////////////////////////////////////////////////////////////
// Kernel declarations
///////////////////////////////////////////////////////////////////////////////

__global__ void CopyGrid(float *destGrid, float *srcGrid, const int bufferSize);
__global__ void Initialize(float *grid, const int bufferSize, const float value);
__global__ void ApplySmokeGenerator(float *grid, const int x, const int y, const float generatorValue, const int gridWidth);
__global__ void IterateSimulation(float *destGrid, float *srcGrid, const int gridDimX, const int gridDimY,
                                  const float advectionCoeff, const float eddyCoeffX, const float eddyCoeffY);
__global__ void IterateSimulationBoundary(float *destGrid, float *srcGrid, const int gridDimX, const int gridDimY,
                                          const float advectionAndEddyXCoeff, const float eddyCoeffY);


int main()
{
    InputConfig config = GetInputConfig(CONFIG_FILENAME);

    // there are a few coefficients that can be caluclated ahead of time
    // the coefficient of the "advection term"
    const float advectionCoefficient =
        (-1.0f * config.advectionConstant * config.timestep) /
        (2 * config.gridCellSizeX);
    
    // the coefficient of the "X eddy term"
    const float eddyCoefficientX = 
        (config.eddyDiffusivityX * config.timestep) / 
        (config.gridCellSizeX * config.gridCellSizeX);

    // combining these two terms is slightly more efficient
    const float advectionAndEddyXCoeff = advectionCoefficient + eddyCoefficientX;
    
    // and the coefficient of the "Y eddy term"
    const float eddyCoefficientY = 
        (config.eddyDiffusivityY * config.timestep) / 
        (config.gridCellSizeY * config.gridCellSizeY);

    const int gridBufferSize = config.gridCellsX * config.gridCellsY;
    std::cout << "Using gridBufferSize of " << gridBufferSize << "\n";

    // generate CPU buffers (for output purposes)
    std::vector<float> cpuGrid(gridBufferSize, 0.0f);
    cimg_library::CImg<uint16_t> outputImg(config.gridCellsX, config.gridCellsY, 1, 1, 0);
    cimg_library::CImgDisplay outputImgDisp(outputImg, "out");

    // generate GPU buffers
    float *d_gridA;
    float *d_gridB;
    HANDLE_ERROR(hipMalloc(&d_gridA, gridBufferSize * sizeof(float)));
    HANDLE_ERROR(hipMalloc(&d_gridB, gridBufferSize * sizeof(float)));

    // initialize all grid cells to having zero smoke concentration
    HANDLE_ERROR(hipMemset(d_gridA, 0, gridBufferSize * sizeof(float)));
    HANDLE_ERROR(hipMemset(d_gridB, 0, gridBufferSize * sizeof(float)));

    dim3 block1D(BLOCK_SIZE, 1, 1);
    dim3 grid1D((gridBufferSize + (block1D.x - 1)) / block1D.x, 1, 1);

    dim3 block2D(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid2D((config.gridCellsX + (block2D.x - 1)) / block2D.x,
                (config.gridCellsY + (block2D.y - 1)) / block2D.y,
                1);
   
    //Initialize<<<grid1D, block1D>>>(d_gridA, gridBufferSize, 100.0f);
    //Initialize<<<grid1D, block1D>>>(d_gridB, gridBufferSize, 100.0f);

    // apply initial condiiton
    // B will be the backbuffer in the first iteration of the loop,
    // which is the buffer that we are pulling values from to caluclate the new (front) buffer
    ApplySmokeGenerator<<<grid1D, block1D>>>(
        d_gridB,
        config.smokeGenLocationX,
        config.smokeGenLocationY,
        config.smokeGenValue,
        config.gridCellsX);

    float *d_grids[2] = { d_gridA, d_gridB };
    const char gridNames[2] = {'A', 'B'}; 
    int currentGrid = 0;
    float t = 0.0f;

    while (t <= config.totalRuntime)
    {
        // front grid is the (n+1)th timestep
        // back grid is the nth timestep
        float *frontGrid = currentGrid % 2 == 0 ? d_gridA : d_gridB;
        float *backGrid  = currentGrid % 2 == 0 ? d_gridB : d_gridA;

        // apply iteration of the function
        IterateSimulationBoundary<<<grid2D, block2D>>>(
            frontGrid,
            backGrid,
            config.gridCellsX,
            config.gridCellsY,
            advectionAndEddyXCoeff,
            eddyCoefficientY);

        // reapply "initial" condition
        // this is applied every cycle since we are modelling a point source of smoke
        // that is - our source has a constant concentration value in time
        ApplySmokeGenerator<<<grid1D, block1D>>>(
            frontGrid,
            config.smokeGenLocationX,
            config.smokeGenLocationY,
            config.smokeGenValue,
            config.gridCellsX);


        // print grid for debug purposes every 10th iteration
        if ((int) t % 10 == 0 && (t - ((int) t)) <= 0.1f)
        {
        HANDLE_ERROR(hipMemcpy(cpuGrid.data(), frontGrid,
                               gridBufferSize * sizeof(float), hipMemcpyDeviceToHost));

        for (int j = config.gridCellsY - 1; j >= 0; j--)
        {
            //std::cout << "[ ";
            for (int i = 0; i < config.gridCellsX; i++)
            {
                //std::cout << cpuGrid[(j * config.gridCellsX) + i] << " ";
                float colorIntensity = cpuGrid[(j * config.gridCellsX) + i] / static_cast<float>(config.smokeGenValue);
                if (colorIntensity > 1.0f)
                    colorIntensity = 1.0f;
                uint16_t intIntensity = colorIntensity * 65535;
                outputImg(i, config.gridCellsY - j - 1, 0, 0) = intIntensity;
            }
            //std::cout << "]\n";
        }
        std::cout << "t=" << t + config.timestep << std::endl;

        // debug visualization
        outputImgDisp.assign(outputImg, "out");
        outputImgDisp.wait(200);
        }

        // swap buffers
        CopyGrid<<<grid1D, block1D>>>(backGrid, frontGrid, gridBufferSize);
        currentGrid = (currentGrid + 1) % 2;

        t += config.timestep;
    }

    return 0;
}

// 1D operation
__global__ void CopyGrid(float *destGrid, float *srcGrid, const int bufferSize)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < bufferSize)
    {
        destGrid[tid] = srcGrid[tid];
    }
}

__global__ void Initialize(float *grid, const int bufferSize, const float value)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid >= bufferSize) return;

    grid[tid] = value;
}

// note that our gridding scheme is
// x horizontal, y vertical such that the indices are...
// (gridDimY - 1)
// ...
// 2
// 1
// 0 1 2 ... (gridDimX - 1)
// so just as you would expect!
__global__ void IterateSimulation(float *destGrid, float *srcGrid, const int gridDimX, const int gridDimY,
                                  const float advectionCoeff, const float eddyCoeffX, const float eddyCoeffY)
{
    // function should be called with blocks of 32x32, for now
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // don't handle values outside the grid
    if (x >= gridDimX || y >= gridDimY)
        return;

    // boundary conditions must be handled differently
    if (x == 0 || x == gridDimX - 1 || y == 0 || y == gridDimY - 1)
        return;

    int cellIdx = (y * gridDimX) + x;
    int cellIdxAbove = ((y + 1) * gridDimX) + x;
    int cellIdxBelow = ((y - 1) * gridDimX) + x;

    float rawValue = (srcGrid[cellIdx]) +
        (advectionCoeff * (srcGrid[cellIdx + 1] - srcGrid[cellIdx - 1])) +                          // Leelossy approx
        (eddyCoeffX * (srcGrid[cellIdx + 1] - srcGrid[cellIdx - 1])) +                              // Leelossy approx
        (eddyCoeffY * (srcGrid[cellIdxAbove] + srcGrid[cellIdxBelow] - (2 * srcGrid[cellIdx])));    // 2nd deriv. approx
    
    // negative concentration doesn't make much sense physically
    destGrid[cellIdx] = rawValue > 0.0f ? rawValue : 0.0f;
}

__global__ void IterateSimulationBoundary(float *destGrid, float *srcGrid, const int gridDimX, const int gridDimY,
                                          const float advectionAndEddyXCoeff, const float eddyCoeffY)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // don't handle values outside the grid
    if (x >= gridDimX || y >= gridDimY)
        return;

    int cellIdx = (y * gridDimX) + x;
    int cellIdxRight = (y * gridDimX) + x + 1;
    int cellIdxLeft  = (y * gridDimX) + x - 1;
    int cellIdxAbove = ((y + 1) * gridDimX) + x;
    int cellIdxBelow = ((y - 1) * gridDimX) + x;

    // fill in cells that would be out of bounds with zero
    float cellValue = srcGrid[cellIdx];
    float cellValueRight = 0.0f;
    float cellValueLeft  = 0.0f;
    float cellValueAbove = 0.0f;
    float cellValueBelow = 0.0f;

    if (x == 0)
    {
        cellValueRight = srcGrid[cellIdxRight];
    }
    else if (x == gridDimX - 1)
    {
        cellValueLeft = srcGrid[cellIdxLeft];
    }
    else
    {
        cellValueRight = srcGrid[cellIdxRight];
        cellValueLeft = srcGrid[cellIdxLeft];
    }
    
    if (y == 0)
    {
        cellValueAbove = srcGrid[cellIdxAbove];
    }
    else if (y == gridDimY - 1)
    {
        cellValueBelow = srcGrid[cellIdxBelow];
    }
    else
    {
        cellValueBelow = srcGrid[cellIdxBelow];
        cellValueAbove = srcGrid[cellIdxAbove];
    }

    float advectionStencil = cellValueRight - cellValueLeft;
    
    float rawValue = (cellValue) +
        (advectionAndEddyXCoeff * advectionStencil) + // Leelossy approx (combined 1st and 2nd terms)
        (eddyCoeffY * (cellValueAbove + cellValueBelow - (2 * cellValue))); // 2nd deriv. approx

    // negative concentration doesn't make much sense physically
    destGrid[cellIdx] = rawValue > 0.0f ? rawValue : 0.0f;
}

__global__ void ApplySmokeGenerator(float *grid, const int x, const int y, const float generatorValue, const int gridWidth)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        grid[(y * gridWidth) + x] = generatorValue;
    }
}



// simple, non-generalizable parser
const InputConfig GetInputConfig(const std::string& inputFilename)
{
    InputConfig config;
    std::ifstream file(inputFilename);
    std::string line;

    while (std::getline(file, line))
    {
        // not-totally-expected behavior: strings starting in whitespace will return an empty string for the first token
        auto tokens = WhitespaceTokenizer(line);

        // this should never be the case, but I'll write it just in case
        if (tokens.size() == 0) continue;

        // empty line
        if (tokens.size() == 1 && tokens[0] == "")
            continue;

        // now we have the option of ignoring the initial whitespace
        if (tokens.size() > 1 && tokens[0] == "")
            tokens.erase(tokens.begin());

        // ignore comments
        if (tokens[0][0] == COMMENT_INDICATOR)
            continue;

        std::string key = "";
        std::string value0 = "";
        std::string value1 = "";

        // parse content lines
        // anything that isn't valid should theoretically be ignored
        if (*(tokens[0].end() - 1) == KEY_VALUE_DELIMITER)
        {
            if (tokens.size() < 2) continue;

            key = {tokens[0].begin(), tokens[0].end() - 1}; // truncate the :

            std::cout << "[key detected]: " << key << std::endl;
            if (*(tokens[1].end() - 1) == TUPLE_DELIMITER)
            {
                if (tokens.size() < 3) continue;

                value0 = {tokens[1].begin(), tokens[1].end() - 1};
                value1 = tokens[2];

                std::cout << "[first value must be]: " << value0 << std::endl;
                std::cout << "[found second value]: " << value1 << std::endl;
            }
            else
            {
                value0 = tokens[1];
                std::cout << "[first value must be]: " << value0 << std::endl;
            }
        }

        // parse keys in a dreadful if/else chain
        if (key == KEY_GRID_CELLS)
        {
            config.gridCellsX = std::stoi(value0);
            config.gridCellsY = std::stoi(value1);
        }
        else if (key == KEY_GRID_SIZE)
        {
            config.gridCellSizeX = std::stof(value0);
            config.gridCellSizeY = std::stof(value1);
        }
        else if (key == KEY_TIMESTEP)
        {
            config.timestep = std::stof(value0);
        }
        else if (key == KEY_RUNTIME)
        {
            config.totalRuntime = std::stof(value0);
        }
        else if (key == KEY_SMOKEGEN_LOC)
        {
            config.smokeGenLocationX = std::stoi(value0);
            config.smokeGenLocationY = std::stoi(value1);
        }
        else if (key == KEY_SMOKEGEN_VAL)
        {
            config.smokeGenValue = std::stof(value0);
        }
        else if (key == KEY_ADVECTION)
        {
            config.advectionConstant = std::stof(value0);
        }
        else if (key == KEY_EDDY)
        {
            config.eddyDiffusivityX = std::stof(value0);
            config.eddyDiffusivityY = std::stof(value1);
        }
        // else continue
    }

    return config;
}

// shout out to StackOverflow
// https://stackoverflow.com/questions/9435385/split-a-string-using-c11
const std::vector<std::string> WhitespaceTokenizer(const std::string& line)
{
    std::regex re("\\s+");
    std::sregex_token_iterator first {line.begin(), line.end(), re, -1};
    std::sregex_token_iterator last;
    return {first, last}; // this is really neat
}

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////

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
    int gridCellSizeX;
    int gridCellSizeY;

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


const InputConfig GetInputConfig(const std::string& inputFilename);
const std::vector<std::string> WhitespaceTokenizer(const std::string& line);

int main()
{
    InputConfig config = GetInputConfig(CONFIG_FILENAME);

    // there are a few coefficients that can be caluclated ahead of time
    // the coefficient of the "advection term"
    const float advectionCoefficient =
        (-1.0f * config.advectionConstant * config.timestep) /
        (2 * config.gridCellSizeX);
    
    // the coefficient of the "X eddy term"
    const float eddyCoefficentX = 
        (config.eddyDiffusivityX * config.timestep) / 
        (config.gridCellSizeX * config.gridCellSizeX);
    
    // and the coefficient of the "Y eddy term"
    const float eddyCoefficientY = 
        (config.eddyDiffusivityY * config.timestep) / 
        (config.gridCellSizeY * config.gridCellSizeY);

    // TODO: generate CPU buffers (for output purposes)

    // generate GPU buffers
    const int gridBufferSize = config.gridCellsX * config.gridCellsY;
    std::cout << "Using gridBufferSize of " << gridBufferSize << "\n";

    float *d_gridA;
    float *d_gridB;
    HANDLE_ERROR(hipMalloc(&d_gridA, gridBufferSize * sizeof(float)));
    HANDLE_ERROR(hipMalloc(&d_gridB, gridBufferSize * sizeof(float)));

    float t = 0.0f;
    while (t <= config.totalRuntime)
    {
        std::cout << "Timestep: " << t << "\n";
        // do stuff
        t += config.timestep;
    }

    return 0;
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
            config.gridCellSizeX = std::stoi(value0);
            config.gridCellSizeY = std::stoi(value1);
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

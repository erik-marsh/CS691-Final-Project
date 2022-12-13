#include <iostream>
#include <string>

const std::string CONFIG_FILENAME = "input.conf";

constexpr char NAME_VALUE_DELIMITER = ':';
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

const InputConfig GetInputConfig(const std::string& inputFilename);

int main()
{
    return 0;
}

const InputConfig GetInputConfig(const std::string& inputFilename)
{
    InputConfig config;

    return config;
}
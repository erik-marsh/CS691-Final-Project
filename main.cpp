#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

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
const std::vector<std::string> WhitespaceTokenizer(const std::string& line);

int main()
{
    GetInputConfig(CONFIG_FILENAME);
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
        // if (line[0] == COMMENT_INDICATOR)
        //     continue;
        
        auto tokens = WhitespaceTokenizer(line);
        // not-totally-expected behavior: strings starting in whitespace will return an empty string for the first token

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
        // anything that isn't valid should be ignored
        if (*(tokens[0].end() - 1) == NAME_VALUE_DELIMITER)
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
#ifndef NEXUSMINER_CONFIG_VALIDATOR_HPP
#define NEXUSMINER_CONFIG_VALIDATOR_HPP

#include <string>
#include <vector>

namespace nexusminer
{
namespace config
{
struct Validator_error
{
    std::string m_field;
    std::string m_message;
};

class Validator
{
public:

    Validator();

    bool check(std::string const& config_file);
    std::string get_check_result() const;

private:

    std::vector<Validator_error> m_mandatory_fields;
    std::vector<Validator_error> m_optional_fields;

};

}
}
#endif

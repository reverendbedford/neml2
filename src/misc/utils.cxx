#include "misc/utils.h"

namespace neml2
{
namespace utils
{
std::string
indentation(int level, int indent)
{
  std::stringstream ss;
  std::string space(indent, ' ');
  for (int i = 0; i < level; i++)
    ss << space;
  return ss.str();
}
} // namespace utils
} // namespace neml2

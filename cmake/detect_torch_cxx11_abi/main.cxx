#include <c10/util/Exception.h>

int
main()
{
  // This is the simplest method that differs before and after the CXX11 ABI change
  throw c10::Error("", "");
  return 0;
}

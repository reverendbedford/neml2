#pragma once

#include <sstream>
#include <iostream>

class NEMLException : public std::exception
{

public:
  NEMLException(char * msg)
    : _message(msg)
  {
  }

  char * what() { return _message; }

private:
  char * _message;
};

/// Throws an exception with given message if the assertion fails. Variadic template is used here so
/// that we can pass in any number of arguments of any type.
template <typename... Args>
void neml_assert(bool assertion, Args &&... args);

/// Similar to assert, but is only effective in debug modes
template <typename... Args>
void neml_assert_dbg(bool assertion, Args &&... args);

/// Internal methods not intended for the public
namespace internal
{
template <typename T, typename... Args>
void stream_all(std::ostringstream & ss, T && val, Args &&... args);

void stream_all(std::ostringstream & ss);
} // namespace internal

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementations for templated methods below
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename... Args>
void
neml_assert(bool assertion, Args &&... args)
{
  if (!assertion)
  {
    std::ostringstream oss;
    internal::stream_all(oss, std::forward<Args>(args)...);
    throw NEMLException(oss.str().data());
  }
}

template <typename... Args>
void
neml_assert_dbg([[maybe_unused]] bool assertion, [[maybe_unused]] Args &&... args)
{
#ifndef NDEBUG
  neml_assert(assertion, args...);
#endif
}

namespace internal
{
template <typename T, typename... Args>
void
stream_all(std::ostringstream & ss, T && val, Args &&... args)
{
  ss << val;
  stream_all(ss, std::forward<Args>(args)...);
}
} // namespace internal

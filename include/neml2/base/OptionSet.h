// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/misc/error.h"
#include "neml2/misc/types.h"
#include "neml2/misc/parser_utils.h"

#include <map>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
#include <memory>

namespace neml2
{
/**
 * Helper functions for printing scalar, vector, vector<vector>. Called from
 * OptionSet::Option<T>::print(...).
 */
template <typename P>
void print_helper(std::ostream & os, const P *);

template <typename P>
void print_helper(std::ostream & os, const std::vector<P> *);

template <typename P>
void print_helper(std::ostream & os, const std::vector<std::vector<P>> *);

/**
 * @brief A custom map-like data structure. The keys are strings, and the values can be
 * nonhomogeneously typed.
 *
 */
class OptionSet
{
public:
  OptionSet() = default;

  /// Copy constructor. Deep copy
  OptionSet(const OptionSet &);

  /// Construct from variadic key-value pairs
  template <typename T, typename... Args>
  OptionSet(const std::pair<std::string, T> & kv, const Args &... kvs)
  {
    fill_kv(kv, kvs...);
  }

  virtual ~OptionSet() = default;

  /// Assignment operator. Deep copy
  virtual OptionSet & operator=(const OptionSet & source);

  /**
   * Addition/Assignment operator.  Inserts copies of all options
   * from \p source.  Any options of the same name already in \p
   * this are replaced. Deep copy.
   */
  virtual OptionSet & operator+=(const OptionSet & source);

  /**
   * Addition operator.  Merge copies of all options
   * from \p source into \p this.  Any options of the same name already in \p
   * this are replaced. Deep copy. Non-commutative.
   */
  virtual OptionSet operator+(const OptionSet & source);

  /**
   * \returns \p true if an option of type \p T
   * with a specified name exists, \p false otherwise.
   *
   * If RTTI has been disabled then we return \p true
   * if an option of specified name exists regardless of its type.
   */
  template <typename T>
  bool contains(const std::string &) const;

  /**
   * \returns A constant reference to the specified option
   * value.  Requires, of course, that the option exists.
   */
  template <typename T>
  const T & get(const std::string &) const;

  /**
   * \returns A writable reference to the specified option.
   * This method will create the option if it does not exist,
   * so it can be used to define options which will later be
   * accessed with the \p get() member.
   */
  template <typename T>
  T & set(const std::string &);

  /// \returns The total number of options
  std::size_t size() const { return _values.size(); }

  /// Clear internal data structures & frees any allocated memory.
  virtual void clear();

  /// Print the contents.
  void print(std::ostream & os = std::cout) const;

  /**
   * Abstract definition of an option value.
   */
  class Value
  {
  public:
    virtual ~Value() = default;

    /**
     * String identifying the type of option stored.
     * Must be reimplemented in derived classes.
     */
    virtual std::string type() const = 0;

    /**
     * Prints the option value to the specified stream.
     * Must be reimplemented in derived classes.
     */
    virtual void print(std::ostream &) const = 0;

    /**
     * Clone this value.  Useful in copy-construction.
     * Must be reimplemented in derived classes.
     */
    virtual std::unique_ptr<Value> clone() const = 0;
  };

  /**
   * Concrete definition of an option value
   * for a specified type
   */
  template <typename T>
  class Option : public Value
  {
  public:
    Option() = default;

    Option(const T & value)
      : _value(value)
    {
    }

    /**
     * \returns A read-only reference to the option value
     */
    const T & get() const { return _value; }

    /**
     * \returns A writable reference to the option value
     */
    T & set() { return _value; }

    virtual std::string type() const;

    virtual void print(std::ostream &) const;

    virtual std::unique_ptr<Value> clone() const;

  private:
    /// Stored option value
    T _value;
  };

  /// The type of the map that we store internally
  typedef std::map<std::string, std::unique_ptr<Value>, std::less<>> map_type;

  /// Option map iterator
  typedef map_type::iterator iterator;

  /// Constant option map iterator
  typedef map_type::const_iterator const_iterator;

  /// Iterator pointing to the beginning of the set of options
  iterator begin();

  /// Iterator pointing to the beginning of the set of options
  const_iterator begin() const;

  /// Iterator pointing to the end of the set of options
  iterator end();

  /// Iterator pointing to the end of the set of options
  const_iterator end() const;

protected:
  void fill_kv() {}

  template <typename T, typename... Args>
  void fill_kv(const std::pair<std::string, T> & kv, const Args &... kvs)
  {
    _values.emplace(kv.first, std::make_unique<Option<T>>(kv.second));
    fill_kv(kvs...);
  }

  /// Data structure to map names with values
  map_type _values;
};

template <typename T>
std::string
OptionSet::Option<T>::type() const
{
  return utils::demangle(typeid(T).name());
}

// LCOV_EXCL_START
template <typename T>
void
OptionSet::Option<T>::print(std::ostream & os) const
{
  print_helper(os, static_cast<const T *>(&_value));
}
// LCOV_EXCL_STOP

template <typename T>
std::unique_ptr<OptionSet::Value>
OptionSet::Option<T>::clone() const
{
  auto copy = std::make_unique<Option<T>>();
  copy->_value = this->_value;
  return copy;
}

std::ostream & operator<<(std::ostream & os, const OptionSet & p);

template <typename T>
bool
OptionSet::contains(const std::string & name) const
{
  OptionSet::const_iterator it = _values.find(name);
  if (it != _values.end())
    if (dynamic_cast<const Option<T> *>(it->second.get()))
      return true;
  return false;
}

template <typename T>
const T &
OptionSet::get(const std::string & name) const
{
  neml_assert(this->contains<T>(name),
              "ERROR: no option named \"",
              name,
              "\" found.\n\nKnown options:\n",
              *this);

  auto ptr = dynamic_cast<Option<T> *>(_values.at(name).get());
  return ptr->get();
}

template <typename T>
T &
OptionSet::set(const std::string & name)
{
  if (!this->contains<T>(name))
    _values[name] = std::make_unique<Option<T>>();
  auto ptr = dynamic_cast<Option<T> *>(_values[name].get());
  return ptr->set();
}

// LCOV_EXCL_START
template <typename P>
void
print_helper(std::ostream & os, const P * option)
{
  os << *option;
}

template <>
inline void
print_helper(std::ostream & os, const char * option)
{
  // Specialization so that we don't print out unprintable characters
  os << static_cast<int>(*option);
}

template <>
inline void
print_helper(std::ostream & os, const unsigned char * option)
{
  // Specialization so that we don't print out unprintable characters
  os << static_cast<int>(*option);
}

template <typename P>
void
print_helper(std::ostream & os, const std::vector<P> * option)
{
  for (const auto & p : *option)
    os << p << " ";
}

template <typename P>
void
print_helper(std::ostream & os, const std::vector<std::vector<P>> * option)
{
  for (const auto & pv : *option)
    for (const auto & p : pv)
      os << p << " ";
}
// LCOV_EXCL_STOP
} // namespace neml2

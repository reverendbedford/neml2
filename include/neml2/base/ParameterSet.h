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
// We should eventually get rid of all of these
typedef std::pair<std::string, std::string> KS;
typedef std::pair<std::string, std::vector<std::string>> KVS;
typedef std::pair<std::string, std::vector<std::vector<std::string>>> KVVS;
typedef std::pair<std::string, Real> KR;
typedef std::pair<std::string, std::vector<Real>> KVR;
typedef std::pair<std::string, std::vector<std::vector<Real>>> KVVR;
typedef std::pair<std::string, bool> KB;
typedef std::pair<std::string, std::vector<bool>> KVB;
typedef std::pair<std::string, std::vector<std::vector<bool>>> KVVB;
typedef std::pair<std::string, unsigned int> KU;
typedef std::pair<std::string, std::vector<unsigned int>> KVU;
typedef std::pair<std::string, std::vector<std::vector<unsigned int>>> KVVU;
typedef std::pair<std::string, TorchSize> KT;
typedef std::pair<std::string, std::vector<TorchSize>> KVT;
typedef std::pair<std::string, std::vector<std::vector<TorchSize>>> KVVT;
typedef std::pair<std::string, int> KI;
typedef std::pair<std::string, std::vector<int>> KVI;
typedef std::pair<std::string, std::vector<std::vector<int>>> KVVI;
typedef std::pair<std::string, LabeledAxisAccessor> KL;
typedef std::pair<std::string, std::vector<LabeledAxisAccessor>> KVL;
typedef std::pair<std::string, std::vector<std::vector<LabeledAxisAccessor>>> KVVL;

/**
 * Helper functions for printing scalar, vector, vector<vector>. Called from
 * ParameterSet::Parameter<T>::print(...).
 */
template <typename P>
void print_helper(std::ostream & os, const P * param);

template <typename P>
void print_helper(std::ostream & os, const std::vector<P> * param);

template <typename P>
void print_helper(std::ostream & os, const std::vector<std::vector<P>> * param);

class ParameterSet
{
public:
  ParameterSet() = default;

  /// Copy constructor. Deep copy
  ParameterSet(const ParameterSet &);

  /// Construct from variadic key-value pairs
  template <typename T, typename... Args>
  ParameterSet(const std::pair<std::string, T> & kv, const Args &... kvs)
  {
    fill_kv(kv, kvs...);
  }

  virtual ~ParameterSet() = default;

  /// Assignment operator. Deep copy
  virtual ParameterSet & operator=(const ParameterSet & source);

  /**
   * Addition/Assignment operator.  Inserts copies of all parameters
   * from \p source.  Any parameters of the same name already in \p
   * this are replaced. Deep copy.
   */
  virtual ParameterSet & operator+=(const ParameterSet & source);

  /**
   * Addition operator.  Merge copies of all parameters
   * from \p source into \p this.  Any parameters of the same name already in \p
   * this are replaced. Deep copy. Non-commutative.
   */
  virtual ParameterSet operator+(const ParameterSet & source);

  /**
   * \returns \p true if a parameter of type \p T
   * with a specified name exists, \p false otherwise.
   *
   * If RTTI has been disabled then we return \p true
   * if a parameter of specified name exists regardless of its type.
   */
  template <typename T>
  bool contains(const std::string &) const;

  /**
   * \returns A constant reference to the specified parameter
   * value.  Requires, of course, that the parameter exists.
   */
  template <typename T>
  const T & get(const std::string &) const;

  /**
   * \returns A writable reference to the specified parameter.
   * This method will create the parameter if it does not exist,
   * so it can be used to define parameters which will later be
   * accessed with the \p get() member.
   */
  template <typename T>
  T & set(const std::string &);

  /// \returns The total number of parameters
  std::size_t size() const { return _values.size(); }

  /// Clear internal data structures & frees any allocated memory.
  virtual void clear();

  /// Print the contents.
  void print(std::ostream & os = std::cout) const;

  /**
   * Abstract definition of a parameter value.
   */
  class Value
  {
  public:
    virtual ~Value() = default;

    /**
     * String identifying the type of parameter stored.
     * Must be reimplemented in derived classes.
     */
    virtual std::string type() const = 0;

    /**
     * Prints the parameter value to the specified stream.
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
   * Concrete definition of a parameter value
   * for a specified type
   */
  template <typename T>
  class Parameter : public Value
  {
  public:
    Parameter() = default;

    Parameter(const T & value)
      : _value(value)
    {
    }

    /**
     * \returns A read-only reference to the parameter value
     */
    const T & get() const { return _value; }

    /**
     * \returns A writable reference to the parameter value
     */
    T & set() { return _value; }

    virtual std::string type() const;

    virtual void print(std::ostream &) const;

    virtual std::unique_ptr<Value> clone() const;

  private:
    /// Stored parameter value
    T _value;
  };

  /// The type of the map that we store internally
  typedef std::map<std::string, std::unique_ptr<Value>, std::less<>> map_type;

  /// Parameter map iterator
  typedef map_type::iterator iterator;

  /// Constant parameter map iterator
  typedef map_type::const_iterator const_iterator;

  /// Iterator pointing to the beginning of the set of parameters
  iterator begin();

  /// Iterator pointing to the beginning of the set of parameters
  const_iterator begin() const;

  /// Iterator pointing to the end of the set of parameters
  iterator end();

  /// Iterator pointing to the end of the set of parameters
  const_iterator end() const;

protected:
  void fill_kv() {}

  template <typename T, typename... Args>
  void fill_kv(const std::pair<std::string, T> & kv, const Args &... kvs)
  {
    _values.emplace(kv.first, std::make_unique<Parameter<T>>(kv.second));
    fill_kv(kvs...);
  }

  /// Data structure to map names with values
  map_type _values;
};

template <typename T>
inline std::string
ParameterSet::Parameter<T>::type() const
{
  return utils::demangle(typeid(T).name());
}

// LCOV_EXCL_START
template <typename T>
inline void
ParameterSet::Parameter<T>::print(std::ostream & os) const
{
  print_helper(os, static_cast<const T *>(&_value));
}
// LCOV_EXCL_STOP

template <typename T>
inline std::unique_ptr<ParameterSet::Value>
ParameterSet::Parameter<T>::clone() const
{
  auto copy = std::make_unique<Parameter<T>>();
  copy->_value = this->_value;
  return copy;
}

inline void
ParameterSet::clear()
{
  _values.clear();
}

inline ParameterSet &
ParameterSet::operator=(const ParameterSet & source)
{
  this->ParameterSet::clear();
  *this += source;
  return *this;
}

inline ParameterSet &
ParameterSet::operator+=(const ParameterSet & source)
{
  for (const auto & [key, value] : source._values)
    _values[key] = value->clone();
  return *this;
}

inline ParameterSet
ParameterSet::operator+(const ParameterSet & source)
{
  ParameterSet ret = *this;
  ret += source;
  return ret;
}

inline ParameterSet::ParameterSet(const ParameterSet & p) { *this = p; }

// LCOV_EXCL_START
inline void
ParameterSet::print(std::ostream & os) const
{
  ParameterSet::const_iterator it = _values.begin();

  while (it != _values.end())
  {
    os << it->first << "\t ";
    it->second->print(os);
    if (++it != _values.end())
      os << '\n';
  }
}

inline std::ostream &
operator<<(std::ostream & os, const ParameterSet & p)
{
  p.print(os);
  return os;
}
// LCOV_EXCL_STOP

template <typename T>
inline bool
ParameterSet::contains(const std::string & name) const
{
  ParameterSet::const_iterator it = _values.find(name);
  if (it != _values.end())
    if (dynamic_cast<const Parameter<T> *>(it->second.get()))
      return true;
  return false;
}

template <typename T>
inline const T &
ParameterSet::get(const std::string & name) const
{
  neml_assert(this->contains<T>(name),
              "ERROR: no parameter named \"",
              name,
              "\" found.\n\nKnown parameters:\n",
              *this);

  auto ptr = dynamic_cast<Parameter<T> *>(_values.at(name).get());
  return ptr->get();
}

template <typename T>
inline T &
ParameterSet::set(const std::string & name)
{
  if (!this->contains<T>(name))
    _values[name] = std::make_unique<Parameter<T>>();
  auto ptr = dynamic_cast<Parameter<T> *>(_values[name].get());
  return ptr->set();
}

inline ParameterSet::iterator
ParameterSet::begin()
{
  return _values.begin();
}

inline ParameterSet::const_iterator
ParameterSet::begin() const
{
  return _values.begin();
}

inline ParameterSet::iterator
ParameterSet::end()
{
  return _values.end();
}

inline ParameterSet::const_iterator
ParameterSet::end() const
{
  return _values.end();
}

// LCOV_EXCL_START
template <typename P>
void
print_helper(std::ostream & os, const P * param)
{
  os << *param;
}

template <>
inline void
print_helper(std::ostream & os, const char * param)
{
  // Specialization so that we don't print out unprintable characters
  os << static_cast<int>(*param);
}

template <>
inline void
print_helper(std::ostream & os, const unsigned char * param)
{
  // Specialization so that we don't print out unprintable characters
  os << static_cast<int>(*param);
}

template <typename P>
void
print_helper(std::ostream & os, const std::vector<P> * param)
{
  for (const auto & p : *param)
    os << p << " ";
}

template <typename P>
void
print_helper(std::ostream & os, const std::vector<std::vector<P>> * param)
{
  for (const auto & pv : *param)
    for (const auto & p : pv)
      os << p << " ";
}
// LCOV_EXCL_STOP
} // namespace neml2

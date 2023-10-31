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

  virtual ~OptionSet() = default;

  /// Assignment operator. Deep copy
  virtual OptionSet & operator=(const OptionSet & source);

  /// A readonly reference to the option set's name
  const std::string & name() const { return _metadata.name; }
  /// A writable reference to the option set's name
  std::string & name() { return _metadata.name; }
  /// A readonly reference to the option set's type
  const std::string & type() const { return _metadata.type; }
  /// A writable reference to the option set's type
  std::string & type() { return _metadata.type; }
  /// A readonly reference to the option set's path
  const std::string & path() const { return _metadata.path; }
  /// A writable reference to the option set's path
  std::string & path() { return _metadata.path; }
  /// A readonly reference to the option set's docstring
  const std::string & doc() const { return _metadata.doc; }
  /// A writable reference to the option set's docstring
  std::string & doc() { return _metadata.doc; }

  /**
   * \returns \p true if an option of type \p T
   * with a specified name exists, \p false otherwise.
   */
  template <typename T>
  bool contains(const std::string &) const;

  /**
   * \returns \p true if an option with a specified name exists, \p false otherwise.
   *
   * We return \p true if an option of specified name exists regardless of its type.
   */
  bool contains(const std::string &) const;

  /// \returns The total number of options
  std::size_t size() const { return _values.size(); }

  /// Clear internal data structures & frees any allocated memory.
  virtual void clear();

  /// Print the contents.
  void print(std::ostream & os = std::cout) const;

  /**
   * Abstract definition of an option.
   */
  class OptionBase
  {
  public:
    virtual ~OptionBase() = default;

    /// A readonly reference to the option's name
    const std::string & name() const { return _metadata.name; }

    /// A readonly reference to the option's type
    const std::string & type() const { return _metadata.type; }

    /// A readonly reference to the option's docstring
    const std::string & doc() const { return _metadata.doc; }

    /// A writable reference to the option's docstring
    std::string & doc() { return _metadata.doc; }

    /// A readonly reference to the option's suppression status
    const bool & suppressed() const { return _metadata.suppressed; }

    /// A writable reference to the option's suppression status
    bool & suppressed() { return _metadata.suppressed; }

    /**
     * Prints the option value to the specified stream.
     * Must be reimplemented in derived classes.
     */
    virtual void print(std::ostream &) const = 0;

    /**
     * Clone this value.  Useful in copy-construction.
     * Must be reimplemented in derived classes.
     */
    virtual std::unique_ptr<OptionBase> clone() const = 0;

  protected:
    /**
     * Metadata associated with this option
     */
    struct Metadata
    {
      /**
       * @brief Name of the option
       *
       * For example, in a HIT input file, this is the field name that appears on the left-hand side
       * ~~~~~~~~~~~~~~~~~python
       * [foo]
       *   type = SomeModel
       *   bar = 123
       * []
       * ~~~~~~~~~~~~~~~~~
       * where "bar" is the option name
       */
      std::string name = "";
      /**
       * @brief Type of the option
       *
       * We use RTTI to determine the type of the option. Most importantly, two options are
       * considered different if they have different types, even if they have the same name. For
       * example, if you specify an option of name "foo" of type `int` as an expected option, later
       * if you attempt to retrieve an option of name "foo" but of type `string`, an exception will
       * be thrown saying that the option does not exist.
       */
      std::string type = "";
      /**
       * @brief Option's doc string
       *
       * When we build the documentation for NEML2, we automatically extract the syntax and convert
       * it to a markdown file. The syntax of NEML2 is just the collection of expected options of
       * all the registered objects. Doxygen will then render the markdown syntax to the target
       * output format, e.g., html, tex, etc. This implies that the docstring can contain anything
       * that the Doxygen's markdown renderer can understand. For more information, see
       * https://www.doxygen.nl/manual/markdown.html
       */
      std::string doc = "";
      /**
       * @brief Whether this option is suppressed
       *
       * By default an option is not suppressed. However, it is sometimes desirable for a derived
       * object to suppress certain option. A suppressed option cannot be modified by the user. It
       * is up to the specific Parser to decide what happens when a user attempts to set a
       * suppressed option, e.g., the parser can choose to throw an exception, print a warning and
       * accepts it, or print a warning and ignores it.
       */
      bool suppressed = false;
    } _metadata;
  };

  /**
   * Concrete definition of an option value
   * for a specified type
   */
  template <typename T>
  class Option : public OptionBase
  {
  public:
    Option(const std::string & name)
      : _value()
    {
      _metadata.name = name;
      _metadata.type = utils::demangle(typeid(T).name());
    }

    /**
     * \returns A read-only reference to the option value
     */
    const T & get() const { return _value; }

    /**
     * \returns A writable reference to the option value
     */
    T & set() { return _value; }

    virtual void print(std::ostream &) const;

    virtual std::unique_ptr<OptionBase> clone() const;

  private:
    /// Stored option value
    T _value;
  };

  /**
   * \returns A constant reference to the specified option value.  Requires, of course, that the
   * option exists.
   */
  template <typename T>
  const T & get(const std::string &) const;

  const OptionBase & get(const std::string &) const;

  /**
   * \returns A writable reference to the specified option value. This method will create the option
   * if it does not exist, so it can be used to define options which will later be accessed with the
   * \p get() member.
   */
  template <typename T>
  T & set(const std::string &);

  OptionBase & set(const std::string &);

  /// The type of the map that we store internally
  typedef std::map<std::string, std::unique_ptr<OptionBase>, std::less<>> map_type;

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
  /**
   * Metadata associated with this option set
   */
  struct Metadata
  {
    /**
     * @brief Name of the option set
     *
     * For example, in a HIT input file, this is the subsection name that appears inside the
     * square brackets
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   type = SomeModel
     *   bar = 123
     * []
     * ~~~~~~~~~~~~~~~~~
     * where "foo" is the name of the option set
     */
    std::string name = "";
    /**
     * @brief Type of the option set
     *
     * For example, in a HIT input file, a special field is reserved for the type of the option
     * set
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   type = SomeModel
     *   bar = 123
     * []
     * ~~~~~~~~~~~~~~~~~
     * where "SomeModel" is the option name. The type is registered to the Registry using
     * register_NEML2_object and its variants.
     */
    std::string type = "";
    /**
     * @brief Path to the option set
     *
     * The path to an option set describes its hierarchy inside the syntax tree parsed by the
     * parser. For example, in a HIT input file, this is the full path to the current option set
     * (excluding its local path contribution)
     * ~~~~~~~~~~~~~~~~~python
     * [foo]
     *   [bar]
     *     [baz]
     *       type = SomeModel
     *       goo = 123
     *     []
     *   []
     * []
     * ~~~~~~~~~~~~~~~~~
     * The option set with name "baz" has path "foo/bar".
     */
    std::string path = "";
    /**
     * @brief Option set's doc string
     *
     * When we build the documentation for NEML2, we automatically extract the syntax and convert
     * it to a markdown file. The syntax of NEML2 is just the collection of expected options of
     * all the registered objects. Doxygen will then render the markdown syntax to the target
     * output format, e.g., html, tex, etc. This implies that the docstring can contain anything
     * that the Doxygen's markdown renderer can understand. For more information, see
     * https://www.doxygen.nl/manual/markdown.html
     */
    std::string doc = "";
  } _metadata;

  /// Data structure to map names with values
  map_type _values;
};

// LCOV_EXCL_START
template <typename T>
void
OptionSet::Option<T>::print(std::ostream & os) const
{
  print_helper(os, static_cast<const T *>(&_value));
}
// LCOV_EXCL_STOP

template <typename T>
std::unique_ptr<OptionSet::OptionBase>
OptionSet::Option<T>::clone() const
{
  auto copy = std::make_unique<Option<T>>(this->name());
  copy->_value = this->_value;
  copy->_metadata = this->_metadata;
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
    _values[name] = std::make_unique<Option<T>>(name);
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

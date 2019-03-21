/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_INI_PARSER_H
#define NEXUS_UTIL_INI_PARSER_H

#include <Util/include/linked_list.h>

typedef char* LPSTR;
typedef const char* LPCSTR;

class IniParserElement
{
public:
  char    value[4096];
  char    name[4096];
};

class IniParserGroup
{
public:
  char    groupName[4096];

  IniParserGroup();
  ~IniParserGroup();

  LinkedListByRef<IniParserElement> elemente;
};

class IniParser
{
  LinkedListByRef<IniParserGroup>  groups;

  void    ClearLine(LPCSTR input, LPSTR output);
  bool    GetValueAndName(LPCSTR line, LPSTR name, LPSTR value);
public:

  IniParser();
  ~IniParser();

  bool    Parse(LPCSTR inhalt);
  void    Clear();

  bool    GetValueAsString(LPCSTR section, LPCSTR value, LPSTR dest, int maxDestSize);
  bool    GetValueAsInteger(LPCSTR section, LPCSTR value, int *dest);
  bool    GetValueAsDouble(LPCSTR section, LPCSTR value, double *dest);
};

#endif

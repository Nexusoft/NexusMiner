/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_ENUMERATOR_H
#define NEXUS_UTIL_ENUMERATOR_H

template <class T>
class EnumerateByValue
{
  typedef struct __wraper__
  {
    struct __wraper__  *next, *prev;
    T ptr;
  }WRAPER, *LPWRAPER;

  LinkedListByValue<T>    *liste;
  LPWRAPER          actWraper;
public:
  EnumerateByValue(LinkedListByValue<T> *list)
  {
    this->liste = list;

    this->actWraper = (LPWRAPER)this->liste->firstElement;
  }

  ~EnumerateByValue()
  {
  }

  void Reset()
  {
    this->actWraper = (LPWRAPER)this->liste->firstElement;
  }

  T GetActValue()
  {
    if(actWraper == NULL)
      return (T)0;

    return actWraper->ptr;
  }

  bool SetToNextElement()
  {
    if(actWraper == NULL)
      return false;

    actWraper = actWraper->next;
    return true;
  }

  bool SetToPrevElement()
  {
    if(actWraper == NULL)
      return false;

    actWraper = actWraper->prev;
    return true;
  }

  bool  IsCurrentValidValue()
  {
    if(actWraper == NULL)
      return false;
    return true;
  }
};

template <class T>
class EnumerateByRef
{
  typedef struct __wraper__
  {
    struct __wraper__  *next, *prev;
    T* ptr;
  }WRAPER, *LPWRAPER;

  LinkedListByRef<T>      *liste;
  LPWRAPER          actWraper;
public:
  EnumerateByRef(LinkedListByRef<T> *list)
  {
    this->liste = list;

    this->actWraper = (LPWRAPER)this->liste->firstElement;
  }

  ~EnumerateByRef()
  {
  }

  void  Reset()
  {
    this->actWraper = (LPWRAPER)this->liste->firstElement;
  }

  T*  GetActValue()
  {
    if(actWraper == NULL)
      return NULL;

    return actWraper->ptr;
  }

  bool SetToNextElement()
  {
    if(actWraper == NULL)
      return false;

    actWraper = actWraper->next;
    return true;
  }

  bool SetToPrevElement()
  {
    if(actWraper == NULL)
      return false;

    actWraper = actWraper->prev;
    return true;
  }

  bool IsCurrentValidValue()
  {
    if(actWraper == NULL)
      return false;
    return true;
  }
};

#define FOREACH_REF(type, pointer, list) for(bool __int__oneTimeLoop__=true;__int__oneTimeLoop__;)for(EnumerateByRef<type> __int__enMine__(&(list));__int__oneTimeLoop__;__int__oneTimeLoop__=false)for(type* pointer=__int__enMine__.GetActValue();__int__enMine__.IsCurrentValidValue() && pointer != NULL;__int__enMine__.SetToNextElement(),pointer=__int__enMine__.GetActValue())
#define FOREACH_VALUE(type, valName, list) for(bool __int__oneTimeLoop__=true;__int__oneTimeLoop__;)for(CEnumerateByValue<type> __int__enMine__(&(list));__int__oneTimeLoop__;__int__oneTimeLoop__=false)for(type valName=__int__enMine__.GetActValue();__int__enMine__.IsCurrentValidValue();__int__enMine__.SetToNextElement(),valName=__int__enMine__.GetActValue())

#endif

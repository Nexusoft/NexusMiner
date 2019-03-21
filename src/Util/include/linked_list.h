/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_LINKEDLIST_H
#define NEXUS_UTIL_LINKEDLIST_H

#include <cstdlib>

template <class T>
class EnumerateByRef;
template <class T>
class EnumerateByValue;

template <class T>
class LinkedListByValue
{
protected:
  typedef struct __wraper__
  {
    struct __wraper__  *next, *prev;
    T ptr;
  }WRAPER, *LPWRAPER;

  volatile LPWRAPER  firstElement, lastElement;
  volatile int    count;
public:
  LinkedListByValue()
  {
    firstElement = NULL;
    lastElement = NULL;
    count = 0;
  };

        ~LinkedListByValue()
  {
    Clear();
  };

  bool AddItemLast(T item)
  {
    int size = sizeof(WRAPER);
    LPWRAPER element = (LPWRAPER)malloc(size);
    if(element == NULL)
      return false;

    memset(element, 0, size);

    element->ptr = item;

    if(firstElement == NULL)
    {
      firstElement = element;
    }else
    {
      lastElement->next = element;
      element->prev = lastElement;
    }
    lastElement = element;

    ++count;
    return true;
  }

  bool AddItemFirst(T item)
  {
    LPWRAPER element = (LPWRAPER)malloc(sizeof(WRAPER));
    if(element == NULL)
      return false;

    memset(element, 0, sizeof(WRAPER));

    element->ptr = item;

    if(lastElement == NULL)
    {
      lastElement = element;
    }else
    {
      firstElement->prev = element;
      element->next = firstElement;
    }

    firstElement = element;
    count++;
    return true;
  }

  void DeleteFirstElement()
  {
    if(firstElement == NULL)
      return;

    LPWRAPER element = firstElement;
    firstElement = firstElement->next;
    if(firstElement != NULL)
      firstElement->prev = NULL;
    else
      lastElement = NULL;

    free(element);
    count--;
  }
  void    DeleteLastElement()
  {
    if(lastElement == NULL)
      return;

    LPWRAPER element = lastElement;
    lastElement = lastElement->prev;
    if(lastElement != NULL)
      lastElement->next = NULL;
    else
      firstElement = NULL;

    free(element);
    count--;
  }


  T      GetFirstItem()
  {
    if(firstElement == NULL)
      return NULL;
    return firstElement->ptr;
  }

  T      GetLastItem()
  {
    if(lastElement == NULL)
      return NULL;
    return lastElement->ptr;
  }

  void    Clear()
  {
    while(true)
    {
      LPWRAPER element = firstElement;
      if(element == NULL)
        break;

      firstElement = firstElement->next;
      free(element);
    }

    firstElement = lastElement = NULL;
    count = 0;
  }

  void    DeleteItem(T item)
  {
    LPWRAPER element = firstElement;
    while(element)
    {
      if(element->ptr == item)
      {
        LPWRAPER prev, next;

        prev = element->prev;
        next = element->next;

        if(prev == NULL)
        {
          firstElement = next;
        }else
        {
          prev->next = next;
        }

        if(next == NULL)
        {
          lastElement = prev;
        }else
        {
          next->prev = prev;
        }

        free(element);
        count--;
        break;
      }
      element = element->next;
    }
  }

  void    MoveItemToBegin(T item)
  {
    LPWRAPER element = firstElement;
    while(element)
    {
      if(element->ptr == item)
      {
        if(element == firstElement)
          break;

        if(element->next)
          element->next->prev = element->prev;
        if(element->prev)
          element->prev->next = element->next;
        element->prev = NULL;
        element->next = firstElement;
        firstElement->prev = element;
        firstElement = element;
        break;
      }

      element = element->next;
    }
  }

  int      GetCount() { return count; };

  T      operator[] (int const& index)
  {
    if(index < 0 || index >= count)
      return NULL;

    EnumerateByValue<T> en(this);
    int counter = 0;
    for(T ptr=en.GetActValue(); en.IsCurrentValidValue();
      en.SetToNextElement(),ptr=en.GetActValue())
    {
      if(counter == index)
        return ptr;
      counter++;
    }
    return NULL;
  }

  friend class EnumerateByValue<T>;
};

template <class T>
class LinkedListByRef
{
protected:
  typedef struct __wraper__
  {
    struct __wraper__  *next, *prev;
    T* ptr;
  }WRAPER, *LPWRAPER;

  volatile LPWRAPER  firstElement, lastElement;
  volatile int    count;
public:
        LinkedListByRef()
  {
    firstElement = NULL;
    lastElement = NULL;
    count = 0;
  }

        ~LinkedListByRef()
  {
    Clear();
  }

  void    Clear()
  {
    Clear(true);
  }

  bool    AddItemLast(T* item)
  {
    int size = sizeof(WRAPER);
    LPWRAPER element = (LPWRAPER)malloc(size);
    if(element == NULL)
      return false;

    memset(element, 0, size);

    element->ptr = item;

    if(firstElement == NULL)
    {
      firstElement = element;
    }else
    {
      lastElement->next = element;
      element->prev = lastElement;
    }
    lastElement = element;

    count++;
    return true;
  }

  bool AddItemFirst(T* item)
  {
    LPWRAPER element = (LPWRAPER)malloc(sizeof(WRAPER));
    if(element == NULL)
      return false;

    memset(element, 0, sizeof(WRAPER));

    element->ptr = item;

    if(lastElement == NULL)
    {
      lastElement = element;
    }else
    {
      firstElement->prev = element;
      element->next = firstElement;
    }

    firstElement = element;
    count++;
    return true;
  }

  void DeleteFirstElement()
  {
    if(firstElement == NULL)
      return;

    LPWRAPER element = firstElement;
    firstElement = firstElement->next;
    if(firstElement != NULL)
      firstElement->prev = NULL;
    else
      lastElement = NULL;

    free(element);
    count--;
  }
  void DeleteLastElement()
  {
    if(lastElement == NULL)
      return;

    LPWRAPER element = lastElement;
    lastElement = lastElement->prev;
    if(lastElement != NULL)
      lastElement->next = NULL;
    else
      firstElement = NULL;

    free(element);
    count--;
  }


  T* GetFirstItem()
  {
    if(firstElement == NULL)
      return NULL;
    return firstElement->ptr;
  }

  T* GetLastItem()
  {
    if(lastElement == NULL)
      return NULL;
    return lastElement->ptr;
  }

  void Clear(bool deleteElements)
  {
    while(true)
    {
      LPWRAPER element = firstElement;
      if(element == NULL)
        break;

      firstElement = firstElement->next;

      if(deleteElements)
      {
        delete element->ptr;
      }

      free(element);
    }

    firstElement = lastElement = NULL;
    count = 0;
  }

  void DeleteItem(T* item)
  {
    LPWRAPER element = firstElement;
    while(element)
    {
      if(element->ptr == item)
      {
        LPWRAPER prev, next;

        prev = element->prev;
        next = element->next;

        if(prev == NULL)
        {
          firstElement = next;
        }else
        {
          prev->next = next;
        }

        if(next == NULL)
        {
          lastElement = prev;
        }else
        {
          next->prev = prev;
        }

        free(element);
        count--;
        break;
      }
      element = element->next;
    }
  }

  void MoveItemToBegin(T* item)
  {
    LPWRAPER element = firstElement;
    while(element)
    {
      if(element->ptr == item)
      {
        if(element == firstElement)
          break;

        if(element->next)
          element->next->prev = element->prev;
        if(element->prev)
          element->prev->next = element->next;
        element->prev = NULL;
        element->next = firstElement;
        firstElement->prev = element;
        firstElement = element;
        break;
      }

      element = element->next;
    }
  }

  int GetCount() { return count; };

  T* operator[] (int const& index)
  {
    if(index < 0 || index >= count)
      return NULL;

    EnumerateByRef<T> en(this);
    int counter = 0;
    for(T* ptr=en.GetActValue(); ptr && en.IsCurrentValidValue();
      en.SetToNextElement(),ptr=en.GetActValue())
    {
      if(counter == index)
        return ptr;
      counter++;
    }
    return NULL;
  }

  friend class EnumerateByRef<T>;
};

#endif

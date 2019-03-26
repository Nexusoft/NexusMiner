/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once

template<int Begin, int End, int Step = 1>
struct Unroller
{
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action)
    {
        action(Begin);
        Unroller<Begin+Step, End, Step>::step(action);
    }
};

template<int End, int Step>
struct Unroller<End, End, Step>
{
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action)
    {
    }
};

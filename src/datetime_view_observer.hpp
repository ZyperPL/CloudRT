#pragma once

class DateTimeViewObserver
{
  public:
    virtual void onNextDayButtonPressed() = 0;
    virtual void onPreviousDayButtonPressed() = 0;
    virtual void onTodayButtonPressed() = 0;
    virtual void onTimeSliderChanged(size_t) = 0;
};

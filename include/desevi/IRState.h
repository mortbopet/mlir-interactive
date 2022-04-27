#pragma once

#include <QString>
#include <assert.h>

struct IRState {
  IRState() {}
  IRState(const QString &state, bool error = false)
      : error(error), state(state) {}

  bool isError() const { return error; }
  const QString &getIR() {
    assert(!isError());
    return state;
  }

  const QString &getError() {
    assert(isError());
    return state;
  }

private:
  bool error = false;
  QString state;
};

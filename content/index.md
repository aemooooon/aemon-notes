---
title: Home
draft: false
tags:
---
```typescript
/**
 * Daily reflection and persistent efforts lead to success.
 * 吾日三省吾身，吾身何患无成？
 * 不积跬步，无以至千里；不积小流，无以成东海。
 */

import cron from "node-cron"
type Action = "Keep practising" | "Keep considering" | "Making notes"

class DailyReflection {
  private dailyActions: Action[] = [
      "Keep practising", 
      "Keep considering", 
      "Making notes"
    ]
  private efforts: number = 0

  public reflectOnSelf(actions: Action[]): string[] {
    return actions.map((action) => `Reflecting on: ${action}`)
  }

  public makeNotes(reflections: string[]): string[] {
    return reflections.map((reflection) => `Note: ${reflection}`)
  }

  public accumulateEfforts(steps: number): void {
    for (let step = 1; step <= steps; step++) {
      this.efforts += step
    }
  }

  public executeDailyRoutine(): void {
    const reflections = this.reflectOnSelf(this.dailyActions)
    const notes = this.makeNotes(reflections)

    console.log("Daily Reflections:")
    reflections.forEach((reflection) => console.log(reflection))

    console.log("\nNotes:")
    notes.forEach((note) => console.log(note))

    this.accumulateEfforts(1000)
    console.log("\nTotal Efforts Accumulated:")
    console.log(this.efforts)

    console.log("\nWisdom:")
    console.log("吾日三省吾身，吾身何患无成？")
    console.log("不积跬步，无以至千里；不积小流，无以成东海。")
    console.log("End of Daily Reflections.")
  }
}

const dailyReflection = new DailyReflection()

cron.schedule("0 0 * * *", () => {
  dailyReflection.executeDailyRoutine()
  console.log("Daily routine executed at:", new Date().toLocaleString())
})
```

- [[Linear Regression]]
- [[Logistic Regression]]
- [[Continuous Variables]]
- [[Cumulative Distribution Function]]
- [[Discrete Variables]]
- [[Multinomial Logistic Regression]]
- [[One Hot Vectors Encoding]]
- [[Probability Density Function]]
- [[Saving and Loading Model]]
- [[Related Metrics]]
- [[Variable Classification]]

![Hua](/static/daughter.jpeg)

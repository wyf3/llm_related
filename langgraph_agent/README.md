```mermaid
graph TD
    START --> create_planner_node
    create_planner_node --> execute_node
    execute_node --> update_planner_node
    update_planner_node --> all_completed
    all_completed -- YES --> report_node
    all_completed -- NO --> execute_node
    report_node --> END
```